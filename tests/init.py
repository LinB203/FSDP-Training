# import argparse
# import os

# import torch
# from transformers import Qwen3ForCausalLM, AutoConfig
# from accelerate import init_empty_weights
# from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# import torch.nn as nn
# from torch.distributed.fsdp import FSDPModule
# from torch.distributed.tensor import distribute_tensor, DTensor


# def load_model(model: FSDPModule, last_model_checkpoint: str):
#     print(f"resume last_model_checkpoint from {last_model_checkpoint}")
#     full_sd = torch.load(
#         last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
#     )
#     meta_sharded_sd = model.state_dict()
#     sharded_sd = {}
#     for param_name, full_tensor in full_sd.items():
#         sharded_meta_param = meta_sharded_sd.get(param_name)
#         sharded_tensor = distribute_tensor(
#             full_tensor,
#             sharded_meta_param.device_mesh,
#             sharded_meta_param.placements,
#         )
#         sharded_sd[param_name] = nn.Parameter(sharded_tensor)
#     # choose `assign=True` since we cannot call `copy_` on meta tensor
#     model.load_state_dict(sharded_sd, strict=False, assign=True)


# def main(args):
#     rank = int(os.environ["LOCAL_RANK"])
#     device_type = torch.accelerator.current_accelerator()
#     device = torch.device(f"{device_type}:{rank}")
#     torch.accelerator.device_index(rank)
#     print(f"Running on rank {rank} on device {device}")

#     backend = torch.distributed.get_default_backend_for_device(device)
#     torch.distributed.init_process_group(backend=backend, device_id=device)

#     torch.manual_seed(0)
#     with init_empty_weights():
#         config = AutoConfig.from_pretrained("/mnt/data/checkpoints/Qwen/Qwen3-8B")
#         model = Qwen3ForCausalLM._from_config(config)  # 只构建结构，不分配权重
#     fsdp_kwargs = {}
#     fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
#         param_dtype=torch.bfloat16,
#         reduce_dtype=torch.float32,
#     )
#     for layer in model.model.layers:
#         fully_shard(layer, **fsdp_kwargs)
#     fully_shard(model, **fsdp_kwargs)
#     load_model(model, "/mnt/data/checkpoints/Qwen/Qwen3-8B/qwen3_8b_merged.pt")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
#     args = parser.parse_args()

#     main(args)


import argparse
import os
import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor


# ---------- 辅助函数 ----------
def load_model(model: FSDPModule, last_model_checkpoint: str):
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
    model.load_state_dict(sharded_sd, strict=False, assign=True)


def run_fsdp_inference(args, prompt, tokenizer):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)  # 设置当前 rank 的 GPU
    device = torch.device(f"cuda:{rank}")

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(0)
    with init_empty_weights():
        config = AutoConfig.from_pretrained(args.model_path)
        model = Qwen3ForCausalLM._from_config(config)
    # model = Qwen3ForCausalLM.from_pretrained(args.model_path)

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }

    for layer in model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    load_model(model, args.merged_ckpt)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
        )[0]
        # 只让 rank0 打印
        # if rank == 0:

    out_ref = torch.load("tmp.pt").to(device)
    diff = (out_ref - outputs).abs()
    print("diff", rank, diff.max(), diff.mean(), torch.allclose(outputs, out_ref))
    return


def run_single_inference(args, prompt, tokenizer):
    device = torch.device("cuda:0")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
        )[0]

        torch.save(outputs, "tmp.pt")
    return


# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/mnt/data/checkpoints/Qwen/Qwen3-8B"
    )
    parser.add_argument(
        "--merged_ckpt",
        type=str,
        default="/mnt/data/checkpoints/Qwen/Qwen3-8B/qwen3_8b_merged.pt",
    )
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "dist"]
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    prompt = "Hello, who are you?"

    # 先跑单卡 baseline
    single_out = run_single_inference(args, prompt, tokenizer)

    fsdp_out = run_fsdp_inference(args, prompt, tokenizer)


if __name__ == "__main__":
    main()

"""
python init.py
torchrun --nproc_per_node=2 init.py --mode dist
"""
