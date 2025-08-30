import os
import random
import torch
import numpy as np
import torch.distributed as dist


def setup_distributed_env():
    # NOTE: adjust backend to 'nccl' for GPU multi-process training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def cleanup_distributed_env():
    dist.destroy_process_group()


def set_seed(
    seed: int,
    global_rank: int,
    device_specific: bool = False,
    deterministic: bool = False,
):
    if device_specific:
        seed += global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
