#!/usr/bin/env python3
"""
verify_2d_equivalence.py

Usage:

# 1) single-gpu run (creates input.pt and single_output.pt)
python tests/verify_2d_equivalence.py --mode single --batch_size 1 --seq_len 128 --seed 1234

# 2) distributed run (use torchrun / torch.distributed launcher)
# adjust --nproc_per_node to number of local GPUs used for TP (e.g. 2)

torchrun --nproc_per_node=2 tests/verify_2d_equivalence.py --mode dist --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234

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
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor.parallel import parallelize_module

from utils.checkpoint import Checkpointer
from utils.tp_plan import base_tp_plan, tp_plan
from utils.fsdp2_warpper import FSDP2_warpper


def build_model(tp_size: int):
    # model_size_cfg can be provided as dict to make it easy to adjust
    # test
    simple_model_config = ModelArgs(
        dim=4096,
        n_layers=2,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=151936,
        tp_size=tp_size,
    )
    model = Transformer.from_model_args(simple_model_config)
    return model


def single_run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = build_model(args.tp_size)
    torch.save(model.state_dict(), f"{args.result_output}/single_model_ckpt.pt")
    model.train()
    model.to(device)

    # deterministic input
    torch.manual_seed(args.seed)  # ensure input deterministic
    inp = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, foreach=True)
    optimizer.zero_grad()
    out = model(inp)
    out.mean().backward()
    optimizer.step()
    print(f"single out: {out.shape}")
    print(f"single inp: {inp.shape}")
    out_cpu = out.detach().cpu()
    inp_cpu = inp.detach().cpu()

    torch.save(inp_cpu, f"{args.result_output}/input.pt")
    torch.save(out_cpu, f"{args.result_output}/single_output.pt")
    torch.save(model.state_dict(), f"{args.result_output}/single_model_ckpt_updated.pt")
    print(
        f"[single] input saved to {args.result_output}/input.pt, output saved to {args.result_output}/single_output.pt, shape {out_cpu.shape}"
    )


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
    print(f"resume last_model_checkpoint from {last_model_checkpoint}")
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
    device_mesh = init_device_mesh(
        device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    # Build model with deterministic init (same seed as single)
    torch.manual_seed(args.seed)
    model = build_model(tp_size=args.tp_size)
    model.train()  # some wrappers want modules in train, but we'll eval later
    # apply parallelize_module to embeddings, norm, output same as your script
    model = parallelize_module(model, tp_mesh, base_tp_plan)

    # per-layer parallelize plan
    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = tp_plan
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    FSDP2_warpper(dp_mesh, model, TransformerBlock, fp32=True)

    load_model(model, f"{args.result_output}/single_model_ckpt.pt")
    model.train()

    # load input on CPU, then move to local device
    inp_cpu = torch.load(f"{args.result_output}/input.pt")
    inp = inp_cpu.to(device_type)

    # forward
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, foreach=True)
    optimizer.zero_grad()
    local_out = model(inp)
    local_out.mean().backward()
    optimizer.step()

    # convert to cpu tensor
    local_out_cpu = local_out.detach().cpu().float()

    # gather local outputs from all ranks via all_gather_object
    gathered = shard_safe_all_gather(local_out_cpu)  # list of CPU tensors or objects
    print(f"local_out_cpu.shape: {local_out_cpu.shape}")

    # rank0 will attempt to reconstruct and compare with single_output.pt
    if global_rank == 0:
        single_out = torch.load(
            f"{args.result_output}/single_output.pt"
        ).float()  # CPU tensor
        target_shape = single_out.shape
        # try direct equal (maybe each rank has full output)
        # check if any gathered element equals target shape
        reconstructed = None
        for elem in gathered:
            if isinstance(elem, torch.Tensor) and tuple(elem.shape) == tuple(
                target_shape
            ):
                reconstructed = elem
                break

        if reconstructed is None:
            # try concatenation along dims
            # ensure all elements are tensors
            tensors_only = []
            for g in gathered:
                if not isinstance(g, torch.Tensor):
                    raise RuntimeError(
                        "Gathered object is not tensor; can't reconstruct automatically."
                    )
                tensors_only.append(g)
            reconstructed = try_concat_to_match(tensors_only, target_shape)

        if reconstructed is None:
            print(
                "[dist rank0] Failed to reconstruct a tensor that matches single output shape."
            )
            print(
                f"single shape: {target_shape}; gathered shapes: {[None if (not isinstance(g,torch.Tensor)) else g.shape for g in gathered]}"
            )
            return

        # compare
        eq = torch.allclose(reconstructed, single_out, atol=args.atol, rtol=args.rtol)
        max_abs_diff = (reconstructed - single_out).abs().max().item()
        mean_abs_diff = (reconstructed - single_out).abs().mean().item()
        print(
            f"[dist rank0] Comparison result: equal={eq}, max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
        )

    dist.barrier()

    checkpointer = Checkpointer("checkpoints", dcp_api=False)
    model_state_dict = checkpointer._get_full_model_state_dict(model)
    # on rank0, reconstruct and compare
    if global_rank == 0:
        model_state_dict_update = torch.load(
            f"{args.result_output}/single_model_ckpt_updated.pt"
        )
        for name, param in model_state_dict.items():
            if name not in model_state_dict_update:
                print(f"[Warning] {name} not in model_state_dict_update")
                continue

            other_param = model_state_dict_update[name]

            # 对齐 dtype 和 device，避免 allclose 报错
            param = param.to(dtype=torch.float32, device="cpu")
            other_param = other_param.to(dtype=torch.float32, device="cpu")

            eq = torch.allclose(param, other_param, atol=args.atol, rtol=args.rtol)
            diff = (param - other_param).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()

            print(
                f"[{name}] equal={eq}, max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
            )
    dist.barrier()

    # on rank0, reconstruct and compare
    if global_rank == 0:
        model_state_dict_update = torch.load(
            f"{args.result_output}/single_model_ckpt.pt"
        )
        for name, param in model_state_dict.items():
            if name not in model_state_dict_update:
                print(f"[Warning] {name} not in model_state_dict_update")
                continue

            other_param = model_state_dict_update[name]

            # 对齐 dtype 和 device，避免 allclose 报错
            param = param.to(dtype=torch.float32, device="cpu")
            other_param = other_param.to(dtype=torch.float32, device="cpu")

            eq = torch.allclose(param, other_param, atol=args.atol, rtol=args.rtol)
            diff = (param - other_param).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()

            print(
                f"[{name}] equal={eq}, max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
            )
    dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "dist"], required=True)
    p.add_argument("--tp_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
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
