# train_qwen3_tokenizer.py
import os
import argparse
from tqdm import tqdm
import math
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler

# hf libs
from transformers import AutoTokenizer
from datasets import load_dataset

# Import your model definition here (same as original)
from models.model import Transformer, ModelArgs, RMSNorm, TransformerBlock
from utils.checkpoint import Checkpointer
from utils.log_utils import rank_log, get_logger
from utils.fsdp2_warpper import FSDP2_mix_warpper


def setup_distributed_env():
    # NOTE: adjust backend to 'nccl' for GPU multi-process training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def cleanup_distributed_env():
    dist.destroy_process_group()


class BlockDataset(Dataset):
    """Holds fixed-length token blocks as torch tensors."""

    def __init__(self, blocks):
        # blocks: list of lists/arrays of token ids of length seq_len
        self.blocks = [torch.tensor(b, dtype=torch.long) for b in blocks]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        # For causal LM: inputs = blocks, labels = blocks (we won't shift here;
        # assumption: model returns logits aligned to input positions)
        input_ids = self.blocks[idx]
        labels = self.blocks[idx].clone()
        return {"input_ids": input_ids, "labels": labels}


def group_texts(examples, tokenizer, seq_len):
    # flatten and chunk
    all_tokens = []
    for txt in examples["text"]:
        toks = tokenizer(txt, add_special_tokens=True)["input_ids"]
        all_tokens.extend(toks)
    # cut into blocks
    n_blocks = len(all_tokens) // seq_len
    blocks = []
    for i in range(n_blocks):
        start = i * seq_len
        blocks.append(all_tokens[start : start + seq_len])
    return blocks


def prepare_dataloader(
    data_path, tokenizer, seq_len, batch_size, split="train", rank=0, world_size=1
):
    # load small dataset
    ds = load_dataset(data_path, "wikitext-2-raw-v1", split=split)
    ds = ds.filter(lambda ex: ex["text"] is not None and len(ex["text"].strip()) > 0)

    # tokenize & accumulate into blocks (same simple approach)
    all_blocks = []
    buffer = []
    for ex in ds:
        toks = tokenizer(ex["text"], add_special_tokens=True)["input_ids"]
        buffer.extend(toks)
        while len(buffer) >= seq_len:
            block = buffer[:seq_len]
            all_blocks.append(block)
            buffer = buffer[seq_len:]
    # create dataset
    dataset = BlockDataset(all_blocks)

    # Create DistributedSampler when world_size > 1
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
        )

    return dataloader, sampler


def train(args):
    logger = get_logger()
    setup_distributed_env()
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    rank_log(
        global_rank,
        logger,
        f"World size: {world_size}, local_rank: {local_rank}, device: {device}",
    )

    # ---- load tokenizer ----
    model_name = args.qwen_model_name  # e.g. "Qwen/Qwen3-0.6B"
    rank_log(global_rank, logger, f"Loading tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    rank_log(global_rank, logger, f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # ---- create model with tokenizer vocab size ----
    simple_model_config = ModelArgs(
        dim=4096, n_layers=2, n_heads=32, n_kv_heads=8, vocab_size=tokenizer.vocab_size
    )
    model = Transformer.from_model_args(simple_model_config)
    model.gradient_checkpointing = args.gradient_checkpointing
    model.train()
    rank_log(
        global_rank,
        logger,
        f"Model init done. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
    )

    FSDP2_mix_warpper(
        None,
        model,
        TransformerBlock,
        norm_to_fp32=RMSNorm,
        reshard_after_forward=args.reshard_after_forward,
    )

    rank_log(
        global_rank,
        logger,
        f"fully_shard Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
    )

    # ---- prepare optimizer and loss ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # ---- load checkpoint if present ----
    checkpointer = Checkpointer(
        "checkpoints",
        dcp_api=args.dcp_api,
        model=model,
        enable_ema=args.enable_ema,
        decay=args.ema_decay,
        fsdp_resharded=args.reshard_after_forward,
    )
    if checkpointer.last_training_time is not None:
        checkpointer.load_model(model)
        checkpointer.load_optim(model, optimizer)
        rank_log(global_rank, logger, "Resumed from checkpoint.")

    # ---- prepare dataloader using HF dataset ----
    rank_log(
        global_rank,
        logger,
        f"Preparing dataset {args.dataset_split} with seq_len={args.seq_len} ...",
    )
    dataloader, sampler = prepare_dataloader(
        args.data_path,
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        split=args.dataset_split,
        rank=global_rank,
        world_size=world_size,
    )

    # ---- training loop ----
    rank_log(global_rank, logger, "Starting training...")
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # 保证每个 epoch 的 shuffle 不同

        running_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(
                device, non_blocking=True
            )  # shape (B, seq_len)
            labels = batch["labels"].to(device, non_blocking=True)  # shape (B, seq_len)

            # forward
            logits = model(input_ids)  # assume logits or (logits, ...)

            B, L, V = logits.shape
            logits_flat = logits.view(B * L, V)
            labels_flat = labels.view(B * L)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            if args.enable_ema:
                if checkpointer.ema_is_registered:
                    checkpointer.ema_update()
                else:
                    checkpointer.ema_register()

            running_loss += loss.item()
            if global_rank == 0 and batch_idx % args.log_interval == 0:
                avg = running_loss / (args.log_interval if args.log_interval > 0 else 1)
                rank_log(
                    global_rank,
                    logger,
                    f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f} | Avg {avg:.4f}",
                )
                running_loss = 0.0

        if args.enable_ema and checkpointer.ema_is_registered:
            for module in model.modules():
                if isinstance(module, torch.distributed.fsdp.FSDPModule):
                    module.reshard()
            checkpointer.ema_apply_shadow()
            output = model(
                input_ids
            )  # test forward, maybe some log_validation function
            for module in model.modules():
                if isinstance(module, torch.distributed.fsdp.FSDPModule):
                    module.reshard()
            checkpointer.ema_restore()
        checkpointer.save(model, optimizer)

    rank_log(global_rank, logger, "Training complete.")
    cleanup_distributed_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF repo id for Qwen3 tokenizer (example 'Qwen/Qwen3-0.6B')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Salesforce/wikitext",
        help="HF repo id for dataset (example 'Salesforce/wikitext')",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
