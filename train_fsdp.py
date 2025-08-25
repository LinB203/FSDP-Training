import os
import itertools
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torch.distributed as dist

# Import your model definition here
from models.model import Transformer, ModelArgs, RMSNorm, TransformerBlock
from utils.checkpoint import Checkpointer
from utils.log_utils import rank_log, get_logger, verify_min_gpu_count
from utils.fsdp2_warpper import FSDP2_mix_warpper


def setup_distributed_env():
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def train(args):
    logger = get_logger()

    # Adjust rank and device
    setup_distributed_env()
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    device = torch.device(
        f"cuda:{local_rank}"
        if torch.cuda.is_available() and world_size > 1
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # test
    # simple_model_config = ModelArgs(
    #     dim=4096, n_layers=2, n_heads=32, n_kv_heads=8, vocab_size=151936
    # )
    # 7B
    # simple_model_config = ModelArgs(
    #     dim=4096, n_layers=36, n_heads=32, n_kv_heads=8, vocab_size=151936
    #     )
    # 13B
    # simple_model_config = ModelArgs(
    #     dim=5120, n_layers=40, n_heads=40, n_kv_heads=8, vocab_size=151936
    # )
    # 32B
    simple_model_config = ModelArgs(
        dim=8192, n_layers=64, n_heads=64, n_kv_heads=8, vocab_size=151936
    )
    model = Transformer.from_model_args(simple_model_config)
    model.gradient_checkpointing = args.gradient_checkpointing
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

    FSDP2_mix_warpper(None, model, TransformerBlock, norm_to_fp32=RMSNorm)
    rank_log(
        global_rank,
        logger,
        f"fully_shard Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
    )

    rank_log(global_rank, logger, f"Model after parallelization {model=}\n")

    # Create an optimizer for the parallelized and sharded model.
    rank_log(
        global_rank, logger, f"Creating AdamW optimizer with learning rate {args.lr}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is not None:
        checkpointer.load_model(model)
        checkpointer.load_optim(model, optimizer)
        rank_log(global_rank, logger, f"resume model...")

    # Training loop:
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    rank_log(global_rank, logger, "\nStarting 2D training...")
    num_iterations = args.size
    batch_size = args.batch_size

    for epoch in range(args.epochs):
        for batch_idx in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            inp = torch.randint(
                model.vocab_size, (batch_size, args.seq_len), device=device
            )

            output = model(inp)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            if global_rank == 0 and batch_idx % args.log_interval == 0:
                rank_log(
                    global_rank,
                    logger,
                    f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}",
                )

        checkpointer.save(model, optimizer)
    rank_log(global_rank, logger, "2D training successfully completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
