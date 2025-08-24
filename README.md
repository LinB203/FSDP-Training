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
  train_fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 8192 \
  --lr 1e-4

```

## FSDP+TP+SP

TP for MLP, SP for attn_head

```
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_sp_tp_fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 8192 \
  --lr 1e-4 \
  --tp_size 2 --head_sp
```


## Valid

Run single gpu to save model, input and output.

```
python tests/verify_2d_equivalence.py --mode single --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234
```

Run tp & FSDP on 2 GPUs

```
torchrun --nproc_per_node=2 tests/verify_2d_equivalence.py --mode dist --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234
```

Run tp & sp & FSDP on 2 GPUs

```
torchrun --nproc_per_node=2 tests/verify_2d_equivalence.py --mode dist --tp_size 2 --batch_size 1 --seq_len 128 --seed 1234 --head_sp
```

The results:

```
[dist rank0] Comparison result: equal=True, max_abs_diff=4.112720e-06, mean_abs_diff=3.527717e-07
```

## References


### FSDP

https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/FSDP2

https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html


### TP
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/tensor_parallelism
