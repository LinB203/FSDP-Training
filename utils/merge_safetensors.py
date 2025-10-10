import os
import torch
from torch import nn
from safetensors.torch import load_file
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)


def merge_weights_to_pt(class_to_load, model_name_or_path, output_path=None):
    merged_state = class_to_load.from_pretrained(model_name_or_path).state_dict()

    if output_path is not None:
        torch.save(merged_state, output_path)
        print(f"Saved merged model to {output_path}")

    return merged_state


def load_safetensors(model: FSDPModule, last_model_checkpoint: str):
    full_sd = torch.load(
        f"{last_model_checkpoint}",
        mmap=True,
        weights_only=True,
        map_location="cpu",
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
    msg = model.load_state_dict(sharded_sd, strict=False, assign=True)
    return msg


# 用法
if __name__ == "__main__":
    merge_weights_to_pt(
        QwenImageTransformer2DModel,
        "/mnt/data/checkpoints/Qwen/Qwen-Image-Edit-2509/transformer",
        "/mnt/data/checkpoints/Qwen/Qwen-Image-Edit-2509/transformer/merged.pt",
    )
