

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
import os
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.log_utils import rank_log, get_logger
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
    FSDPModule,
)
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from utils.checkpoint import Checkpointer

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

# Simple random dataset
class RandomTokenDataset(Dataset):
    def __init__(self, seq_len, dataset_size, vocab_size):
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # generate random token sequence and target (shifted by 1)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return tokens[:-1], tokens[1:]
    

def train(args):

    tp_size = args.tp_size
    logger = get_logger()

    # understand world topology
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])


    rank_log(global_rank, logger, f"Starting PyTorch 2D (FSDP + TP) example on rank {global_rank}.")
    assert (
        world_size % tp_size == 0
    ), f"World size {world_size} needs to be divisible by TP size {tp_size}"


    # create a sharding plan based on the given world_size.
    dp_size = world_size // tp_size

    device_type = torch.accelerator.current_accelerator().type
    # Create a device mesh with 2 dimensions.
    # First dim is the data parallel dimension
    # Second dim is the tensor parallel dimension.
    device_mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    rank_log(global_rank, logger, f"Device Mesh created: {device_mesh=}")
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()

    model_name = "/mnt/data/checkpoints/Qwen/Qwen3-14B"
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    model._set_gradient_checkpointing(True)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    rank_log(global_rank, logger, f'model.dtype: {model.dtype}')
    rank_log(global_rank, logger, f"init meta Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rank_log(global_rank, logger, f"Total parameters: {total_params:,}")
    rank_log(global_rank, logger, f"Trainable parameters: {trainable_params:,}")

    model = parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )

    for layer_id, transformer_block in enumerate(model.model.layers):
        layer_tp_plan = {
            "input_layernorm": SequenceParallel(),
            "post_attention_layernorm": SequenceParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
        }

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block.mlp,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )


    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

    cpu_offload = False
    mp_policy_fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32,    # 参数都以 float32 送进计算
        reduce_dtype=torch.float32,   # 梯度也用 float32 汇总
        output_dtype=torch.bfloat16, 
    )
    mp_policy_bf16 = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,   # 其余层用 bfloat16
        reduce_dtype=torch.float32,   # 梯度还是 upcast 到 float32
        output_dtype=torch.bfloat16, 
    )
    def shard_module(mod, **fsdp_kwargs):
        if isinstance(mod, (Qwen3RMSNorm)):
            # print(f'here mod: {mod} should be Qwen3RMSNorm')
            return fully_shard(mod, mp_policy=mp_policy_fp32, **fsdp_kwargs)
        else:
            return fully_shard(mod, mp_policy=mp_policy_bf16, **fsdp_kwargs)

    fsdp_kwargs = {
        "reshard_after_forward": False,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    for module in model.modules():
        if isinstance(module, Qwen3RMSNorm):
            shard_module(module, **fsdp_kwargs)
    for module in model.modules():
        if isinstance(module, Qwen3DecoderLayer):
            shard_module(module, **fsdp_kwargs)
    shard_module(model, **fsdp_kwargs)
    rank_log(global_rank, logger, f"fully_shard Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    rank_log(global_rank, logger, f"Model after parallelization {model=}\n")

    # Create an optimizer for the parallelized and sharded model.
    rank_log(global_rank, logger, f"Creating AdamW optimizer with learning rate {args.lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    criterion = torch.nn.CrossEntropyLoss()

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is not None:
        checkpointer.load_model(model)
        checkpointer.load_optim(model, optimizer)
        rank_log(global_rank, logger, f'resume model...')

    # dataset and loader
    # dataset = RandomTokenDataset(seq_len=args.seq_len + 1, dataset_size=args.size, vocab_size=model.vocab_size)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    # loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Training loop:
    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    rank_log(global_rank, logger, "\nStarting 2D training...")
    num_iterations = args.size

    for epoch in range(args.epochs):
        # if sampler:
        #     sampler.set_epoch(epoch)
        for batch_idx in tqdm(range(num_iterations)):
            # seeding with dp_rank to ensure identical inputs for TP groups
            torch.manual_seed(batch_idx + dp_rank)
            inp = torch.randint(model.vocab_size, (1, args.seq_len), device=device_type)

            output = model(inp, use_cache=False)
            loss = output[0].sum()
            loss.backward()
            optimizer.step()

            if global_rank == 0 and batch_idx % args.log_interval == 0:
                rank_log(global_rank, logger, f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

        checkpointer.save(model, optimizer)
    rank_log(global_rank, logger, "2D training successfully completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
