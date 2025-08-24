# train_2d_with_datacache_qwen3.py
"""
2D TP+DP + FSDP test script with data-cache (TP-group gather) behavior.

Design:
- Each rank uses a standard DistributedSampler / DataLoader (DDP-style: each card draws different data).
- For each batch, inside each TP group we do an all_gather to create a datacache containing
  one local batch from each TP-rank in the group.
- We then loop over the datacache entries (tp times), feed each entry to model forward,
  scale loss by 1/tp_size and backward. After the loop, call optimizer.step().
- Tokenizer: Qwen3 (default "Qwen/Qwen3-0.6B").
- Dataset: wikitext-2-raw-v1 (Hugging Face) used for quick LM loss testing.
References:
 - Qwen3 on HF: https://huggingface.co/Qwen/Qwen3-0.6B. :contentReference[oaicite:1]{index=1}
 - wikitext-2-raw-v1 on HF. :contentReference[oaicite:2]{index=2}
"""
import os
import argparse
from tqdm import tqdm
import math
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from transformers import AutoTokenizer
from datasets import load_dataset

from models.model import Transformer, ModelArgs, RMSNorm, TransformerBlock
from utils.checkpoint import Checkpointer
from utils.log_utils import rank_log, get_logger
from utils.tp_plan import base_tp_plan, head_sp_tp_plan, tp_plan
from utils.fsdp2_warpper import FSDP2_mix_warpper


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


# ---------------------------
# utils: tp-group creation (assume consecutive grouping)
# ---------------------------
def make_tp_group(world_size, tp_size, global_rank):
    """
    Create TP-group by consecutive blocks:
    ranks = [0..tp_size-1], [tp_size..2*tp_size-1], ...
    This matches common device mesh layouts like [[0,1],[2,3],...]
    If your rank mapping is different, replace with mesh-based group construction.
    """
    group_start = (global_rank // tp_size) * tp_size
    tp_group_ranks = list(range(group_start, group_start + tp_size))
    return dist.new_group(ranks=tp_group_ranks), tp_group_ranks


# ---------------------------
# Main train function
# ---------------------------
def train(args):

    head_sp = args.head_sp
    tp_size = args.tp_size
    logger = get_logger()

    # understand world topology
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # create a sharding plan based on the given world_size.
    dp_size = world_size // tp_size

    device_type = torch.accelerator.current_accelerator().type
    # Create a device mesh with 2 dimensions.
    # First dim is the data parallel dimension
    # Second dim is the tensor parallel dimension.
    device_mesh = init_device_mesh(
        device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    rank_log(global_rank, logger, f"Device Mesh created: {device_mesh=}")
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()

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

    # build model config & model (vocab_size aligned)
    simple_model_config = ModelArgs(
        dim=4096, n_layers=2, n_heads=32, n_kv_heads=8, vocab_size=151936
    )
    if head_sp:
        assert simple_model_config.n_heads % tp_size == 0
    model = Transformer.from_model_args(simple_model_config)
    model.gradient_checkpointing = True
    model.train()
    rank_log(
        global_rank,
        logger,
        f"init model Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_log(global_rank, logger, f"Total parameters: {total_params:,}")
    rank_log(global_rank, logger, f"Trainable parameters: {trainable_params:,}")

    # parallelize the first embedding and the last linear out projection
    model = parallelize_module(model, tp_mesh, base_tp_plan)
    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = head_sp_tp_plan if head_sp else tp_plan
        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    FSDP2_mix_warpper(dp_mesh, model, TransformerBlock, norm_to_fp32=RMSNorm)
    rank_log(
        global_rank,
        logger,
        f"fully_shard Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
    )
    rank_log(global_rank, logger, f"Model after parallelization {model=}\n")

    # optimizer & checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)
    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is not None:
        checkpointer.load_model(model)
        checkpointer.load_optim(model, optimizer)
        rank_log(global_rank, logger, "Resumed checkpoint.")

    # -------------------------
    # Prepare dataloader (standard DDP sampler per-rank)
    # -------------------------
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

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Create TP group (consecutive block assumption). If your mesh is different, replace this.
    tp_group, tp_group_ranks = make_tp_group(world_size, tp_size, global_rank)
    if global_rank == 0:
        rank_log(global_rank, logger, f"tp_group_ranks example = {tp_group_ranks}")

    # training loop with data-cache in TP group
    for epoch in range(args.epochs):
        # ensure sampler epoch shuffle
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        running_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # each rank gets its own local batch (different samples)
            local_input = batch["input_ids"].to(
                device_type, non_blocking=True
            )  # (B, L)
            local_labels = batch["labels"].to(device_type, non_blocking=True)  # (B, L)

            # pre-alloc gather buffers (each element must be same shape)
            gather_inputs = [torch.empty_like(local_input) for _ in range(tp_size)]
            gather_labels = [torch.empty_like(local_labels) for _ in range(tp_size)]

            # all_gather within TP group: each rank will receive the full datacache
            dist.all_gather(gather_inputs, local_input, group=tp_group)
            dist.all_gather(gather_labels, local_labels, group=tp_group)

            # datacache: list of (input_ids, labels) of length tp_size (same across TP group)
            datacache = list(zip(gather_inputs, gather_labels))

            # zero grads once, then loop over datacache and accumulate grads
            optimizer.zero_grad()
            for k in range(tp_size):
                in_k, lab_k = datacache[k]
                # ensure tensors on correct device (usually already are)
                # in_k = in_k.to(device, non_blocking=True)
                # lab_k = lab_k.to(device, non_blocking=True)

                logits = model(in_k)

                # logits shape (B, L, V)
                B, L, V = logits.shape
                loss = criterion(logits.view(B * L, V), lab_k.view(B * L))
                # normalize by tp_size so final gradient magnitude is consistent
                (loss / tp_size).backward()
                running_loss += loss.item()

            optimizer.step()
            if global_rank == 0 and batch_idx % args.log_interval == 0:
                avg = running_loss / (args.log_interval if args.log_interval > 0 else 1)
                rank_log(
                    global_rank,
                    logger,
                    f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f} | Avg {avg:.4f}",
                )
                running_loss = 0.0

        # save
        checkpointer.save(model, optimizer)

    rank_log(global_rank, logger, "2D training successfully completed!")


# ---------------------------
# CLI entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--head_sp", action="store_true", default=False)
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

# ---------------------------
# Example run (single node 4 GPUs, tp_size=2):
# torchrun --nproc_per_node=4 train_2d_with_datacache_qwen3.py --tp_size 2 --batch_size 4 --epochs 1
# Notes:
# - This script uses DistributedSampler (drop_last=True) so that all_gather works with fixed-shaped tensors.
# - The make_tp_group() function assumes TP groups are consecutive blocks of ranks (0..tp-1, tp..2tp-1, ...).
#   If your device mesh ranks are arranged differently, replace make_tp_group with mesh-based group building.
# ---------------------------
