import functools
import torch
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)


def rgetattr(obj, attr_path):
    return functools.reduce(getattr, attr_path.split("."), obj)


def FSDP2_warpper(dp_mesh, model, main_block=None, fp32=False):
    cpu_offload = False
    mp_policy_fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32,  # 参数都以 float32 送进计算
        reduce_dtype=torch.float32,  # 梯度也用 float32 汇总
        output_dtype=torch.float32,
    )
    mp_policy_bf16 = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,  # 其余层用 bfloat16
        reduce_dtype=torch.bfloat16,  # 梯度还是 upcast 到 float32
        output_dtype=torch.bfloat16,
    )
    fsdp_kwargs = {
        "reshard_after_forward": False,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if main_block is not None:
        for module in model.modules():
            if isinstance(module, main_block):
                fully_shard(
                    module,
                    mp_policy=mp_policy_fp32 if fp32 else mp_policy_bf16,
                    **fsdp_kwargs
                )
    fully_shard(
        model, mp_policy=mp_policy_fp32 if fp32 else mp_policy_bf16, **fsdp_kwargs
    )


def FSDP2_mix_warpper(dp_mesh, model, main_block_to_bf16=None, norm_to_fp32=None):
    cpu_offload = False
    mp_policy_fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32,  # 参数都以 float32 送进计算
        reduce_dtype=torch.float32,  # 梯度也用 float32 汇总
        output_dtype=torch.bfloat16,
    )
    mp_policy_bf16 = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,  # 其余层用 bfloat16
        reduce_dtype=torch.float32,  # 梯度还是 upcast 到 float32
        output_dtype=torch.bfloat16,
    )
    fsdp_kwargs = {
        "reshard_after_forward": False,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if norm_to_fp32 is not None:
        for module in model.modules():
            if isinstance(module, norm_to_fp32):
                fully_shard(module, mp_policy=mp_policy_fp32, **fsdp_kwargs)
    for module in model.modules():
        if main_block_to_bf16 is not None and isinstance(module, main_block_to_bf16):
            fully_shard(module, mp_policy=mp_policy_bf16, **fsdp_kwargs)
    fully_shard(model, mp_policy=mp_policy_bf16, **fsdp_kwargs)


def set_modules_to_forward_prefetch(
    model, block_name="transformer_blocks", num_to_forward_prefetch=2
):
    for i, layer in enumerate(rgetattr(model, block_name)):
        if i >= len(rgetattr(model, block_name)) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            rgetattr(model, block_name)[i + j]
            for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(
    model, block_name="transformer_blocks", num_to_backward_prefetch=2
):
    for i, layer in enumerate(rgetattr(model, block_name)):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            rgetattr(model, block_name)[i - j]
            for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)
