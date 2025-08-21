import os
import itertools
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
    FSDPModule,
)
# Import your model definition here
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from utils.checkpoint import Checkpointer

from utils.log_utils import rank_log, get_logger

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

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 1 else "cuda" if torch.cuda.is_available() else "cpu")

    model_name = "/mnt/data/checkpoints/Qwen/Qwen3-14B"
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    # model.forward = torch.compile(model.forward)
    model._set_gradient_checkpointing(True)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    rank_log(global_rank, logger, f'model.dtype: {model.dtype}')
    rank_log(global_rank, logger, f"init meta Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rank_log(global_rank, logger, f"Total parameters: {total_params:,}")
    rank_log(global_rank, logger, f"Trainable parameters: {trainable_params:,}")

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
        "mesh": None,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is not None:
        checkpointer.load_model(model)
        checkpointer.load_optim(model, optimizer)
        rank_log(global_rank, logger, f'resume model...')

    # dataset and loader
    dataset = RandomTokenDataset(seq_len=args.seq_len + 1, dataset_size=args.size, vocab_size=model.vocab_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)


    # dtype_records = {}
    # def hook_fn(module, inputs, output):
    #     dtype_records[module] = {
    #         'input_dtype':  inputs[0].dtype,
    #         # 'weight_dtype': module.weight.dtype,
    #         'output_dtype': output[0].dtype if isinstance(output, tuple) else output.dtype,
    #     }

    # for mod in model.modules():
    #     if isinstance(mod, (Qwen2RMSNorm)) or isinstance(mod, (Qwen2DecoderLayer)):
    #         print(f'hook to {mod}')
    #         mod.register_forward_hook(hook_fn)


    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, use_cache=False)
            # reshape: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            # loss = criterion(outputs.logits.view(-1, model.vocab_size), targets.view(-1))
            loss = outputs.logits.sum()
            loss.backward()
            optimizer.step()

            if global_rank == 0 and batch_idx % args.log_interval == 0:
                rank_log(global_rank, logger, f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
            # for mod, rec in dtype_records.items():
            #     print(f"{mod.__class__.__name__}:", 
            #         "input:",  rec['input_dtype'], 
            #         # "weight:", rec['weight_dtype'],
            #         "output:", rec['output_dtype'])
        checkpointer.save(model, optimizer)
    cleanup_distributed_env()


def main():
    parser = argparse.ArgumentParser()
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
