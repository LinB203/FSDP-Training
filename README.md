A minimal parallel training example.

## Env
torch ≥ 2.7.1
```
pip install -r requirements.txt
```

## Only FSDP

Using [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) from torch

```
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 1024 \
  --lr 1e-4
```

## FSDP+TP+SP

TP for MLP, SP for all norm layers

```
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tp_fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 1024 \
  --lr 1e-4
```

## TODO

### sp@attn_head

### tp@qkvo_proj

## References


### FSDP

https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/FSDP2

https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html


### TP
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/tensor_parallelism
