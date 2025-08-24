#!/usr/bin/env python3
"""
verify_2d_equivalence.py

Usage:

# 1) single-gpu run (creates input.pt and single_output.pt)
python tests/verify_2d_equivalence.py --mode single --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234

# 2) distributed run (use torchrun / torch.distributed launcher)
# adjust --nproc_per_node to number of local GPUs used for TP (e.g. 2)

torchrun --nproc_per_node=2 tests/verify_2d_equivalence.py --mode dist --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234
torchrun --nproc_per_node=2 tests/verify_2d_equivalence.py --mode dist --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234 --head_sp

The script will print whether outputs match (within atol/rtol) and some diagnostics.
"""
import os
import argparse
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))
from models.model import Transformer, ModelArgs, RMSNorm, TransformerBlock

# imports from your script
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    SequenceParallel,
)

def build_model(tp_size: int, head_sp: bool):
    # model_size_cfg can be provided as dict to make it easy to adjust
    # test
    simple_model_config = ModelArgs(
        dim=4096, n_layers=2, n_heads=32, n_kv_heads=8, vocab_size=151936, head_sp=head_sp, tp_size=tp_size
        )
    model = Transformer.from_model_args(simple_model_config)
    return model

def single_run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = build_model(args.tp_size, args.head_sp)
    model.eval()
    model.to(device)

    # deterministic input
    torch.manual_seed(args.seed)  # ensure input deterministic
    inp = torch.randint(low=0, high=model.vocab_size, size=(args.batch_size, args.seq_len), device=device)

    with torch.no_grad():
        out = model(inp)
    print(f"single out: {out.shape}")
    print(f"single inp: {inp.shape}")
    out_cpu = out.detach().cpu()
    inp_cpu = inp.detach().cpu()

    torch.save(inp_cpu, f"{args.result_output}/input.pt")
    torch.save(out_cpu, f"{args.result_output}/single_output.pt")
    torch.save(model.state_dict(), f"{args.result_output}/single_model_ckpt.pt")
    print(f"[single] input saved to {args.result_output}/input.pt, output saved to {args.result_output}/single_output.pt, shape {out_cpu.shape}")

def try_concat_to_match(tensors_list, target_shape):
    """
    Given a list of CPU tensors (from each rank), try concatenating along each dim
    to see if we can match target_shape. Return the concatenated tensor if matched, else None.
    """
    for dim in range(len(target_shape)):
        try:
            cand = torch.cat(tensors_list, dim=dim)
            if tuple(cand.shape) == tuple(target_shape):
                return cand
        except Exception:
            pass
    return None

def shard_safe_all_gather(local_tensor):
    """
    Gather CPU tensors from all ranks via all_gather_object into a list.
    This avoids shape/equal-size constraints of all_gather.
    """
    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    # all_gather_object will fill `gathered` on each rank with objects from every rank
    dist.all_gather_object(gathered, local_tensor)
    return gathered

def load_model(model: FSDPModule, last_model_checkpoint):
    print(f'resume last_model_checkpoint from {last_model_checkpoint}')
    full_sd = torch.load(
        last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
    )
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, strict=False, assign=True)

def dist_run(args):

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_type = torch.accelerator.current_accelerator().type

    # create device mesh similar to your training script
    tp_size = args.tp_size
    assert world_size % tp_size == 0, "world_size must be divisible by tp_size"
    dp_size = world_size // tp_size
    # mesh dims: (dp, tp)
    device_mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # Build model with deterministic init (same seed as single)
    torch.manual_seed(args.seed)
    model = build_model(tp_size=args.tp_size, head_sp=args.head_sp)
    model.train()  # some wrappers want modules in train, but we'll eval later
    # apply parallelize_module to embeddings, norm, output same as your script
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )

    # per-layer parallelize plan
    for layer_id, transformer_block in enumerate(model.layers):
        if args.head_sp:
            layer_tp_plan = {
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), ),
                    desired_input_layouts=(Replicate(), ),
                ),
                "attention.wq": ColwiseParallel(use_local_output=True),
                "attention.wk": ColwiseParallel(use_local_output=True),
                "attention.wv": ColwiseParallel(use_local_output=True),
                "attention.sp_head": PrepareModuleInput(
                    input_layouts=(Shard(1), Shard(1), Shard(1)),
                    desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
                ),
                "attention.sp_head": PrepareModuleOutput(
                    output_layouts=(Shard(1), ),
                    desired_output_layouts=(Shard(1), ),
                ),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            }
        else:
            layer_tp_plan = {
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), ),
                    desired_input_layouts=(Replicate(), ),
                ),
                "attention.wq": ColwiseParallel(use_local_output=False),
                "attention.wk": ColwiseParallel(use_local_output=False),
                "attention.wv": ColwiseParallel(use_local_output=False),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            }
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )

    # FSDP wrapping similar to your script
    mp_policy_fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    def shard_module(mod, **fsdp_kwargs):
        return fully_shard(mod, mp_policy=mp_policy_fp32, **fsdp_kwargs)

    fsdp_kwargs = {
        "reshard_after_forward": False,
        "mesh": dp_mesh,
    }

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            shard_module(module, **fsdp_kwargs)
    # finally shard top-level model
    shard_module(model, **fsdp_kwargs)

    load_model(model, f"{args.result_output}/single_model_ckpt.pt")
    model.eval()

    # load input on CPU, then move to local device
    inp_cpu = torch.load(f"{args.result_output}/input.pt")
    inp = inp_cpu.to(device_type)

    # forward
    with torch.no_grad():
        local_out = model(inp)

    # convert to cpu tensor
    local_out_cpu = local_out.detach().cpu().float()

    # gather local outputs from all ranks via all_gather_object
    gathered = shard_safe_all_gather(local_out_cpu)  # list of CPU tensors or objects
    print(f"local_out_cpu.shape: {local_out_cpu.shape}")

    # rank0 will attempt to reconstruct and compare with single_output.pt
    if global_rank == 0:
        single_out = torch.load(f"{args.result_output}/single_output.pt").float()  # CPU tensor
        target_shape = single_out.shape
        # try direct equal (maybe each rank has full output)
        # check if any gathered element equals target shape
        reconstructed = None
        for elem in gathered:
            if isinstance(elem, torch.Tensor) and tuple(elem.shape) == tuple(target_shape):
                reconstructed = elem
                break

        if reconstructed is None:
            # try concatenation along dims
            # ensure all elements are tensors
            tensors_only = []
            for g in gathered:
                if not isinstance(g, torch.Tensor):
                    raise RuntimeError("Gathered object is not tensor; can't reconstruct automatically.")
                tensors_only.append(g)
            reconstructed = try_concat_to_match(tensors_only, target_shape)

        if reconstructed is None:
            print("[dist rank0] Failed to reconstruct a tensor that matches single output shape.")
            print(f"single shape: {target_shape}; gathered shapes: {[None if (not isinstance(g,torch.Tensor)) else g.shape for g in gathered]}")
            return

        # compare
        eq = torch.allclose(reconstructed, single_out, atol=args.atol, rtol=args.rtol)
        max_abs_diff = (reconstructed - single_out).abs().max().item()
        mean_abs_diff = (reconstructed - single_out).abs().mean().item()
        print(f"[dist rank0] Comparison result: equal={eq}, max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}")

    dist.barrier()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "dist"], required=True)
    p.add_argument("--tp_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--head_sp", action="store_true", default=False)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--result_output", type=str, default="valid_tmp_output")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.result_output, exist_ok=True)
    if args.mode == "single":
        single_run(args)
    elif args.mode == "dist":
        dist_run(args)
    else:
        raise RuntimeError("unknown mode")

