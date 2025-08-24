"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

import sys
import argparse
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module

from models.model import Transformer, ModelArgs, RMSNorm, TransformerBlock
from utils.checkpoint import Checkpointer
from utils.log_utils import rank_log, get_logger
from utils.tp_plan import base_tp_plan, head_sp_tp_plan, tp_plan
from utils.fsdp2_warpper import FSDP2_mix_warpper


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

    # test
    simple_model_config = ModelArgs(
        dim=4096,
        n_layers=2,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=151936,
        head_sp=head_sp,
        tp_size=tp_size,
    )
    # 7B
    # simple_model_config = ModelArgs(
    #     dim=4096, n_layers=36, n_heads=32, n_kv_heads=8, vocab_size=151936, head_sp=head_sp, tp_size=tp_size
    #     )
    # 13B
    # simple_model_config = ModelArgs(
    #     dim=5120, n_layers=40, n_heads=40, n_kv_heads=8, vocab_size=151936, head_sp=head_sp, tp_size=tp_size
    #     )
    # 32B
    # simple_model_config = ModelArgs(
    #     dim=5120, n_layers=64, n_heads=64, n_kv_heads=8, vocab_size=151936, head_sp=head_sp, tp_size=tp_size
    #     )
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

    # Create an optimizer for the parallelized and sharded model.
    rank_log(
        global_rank, logger, f"Creating AdamW optimizer with learning rate {args.lr}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)

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
            # seeding with dp_rank to ensure identical inputs for TP groups
            torch.manual_seed(batch_idx + dp_rank)
            inp = torch.randint(
                model.vocab_size, (batch_size, args.seq_len), device=device_type
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
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--head_sp", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
