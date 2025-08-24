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

# real dataset
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_fsdp_real.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 8192 \
  --lr 1e-4 \
  --qwen_model_name /mnt/data/checkpoints/Qwen/Qwen3-0.6B \
  --data_path /mnt/data/datasets/Salesforce/wikitext
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


# real dataset
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_fsdp_real.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 8192 \
  --lr 1e-4 \
  --qwen_model_name /mnt/data/checkpoints/Qwen/Qwen3-0.6B \
  --data_path /mnt/data/datasets/Salesforce/wikitext
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

[tok_embeddings.weight] equal=True, max_abs_diff=5.960464e-08, mean_abs_diff=1.981021e-16
[layers.0.attention.wq.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=1.264637e-13
[layers.0.attention.wk.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=2.701494e-13
[layers.0.attention.wv.weight] equal=True, max_abs_diff=1.862645e-09, mean_abs_diff=3.898042e-12
[layers.0.attention.wo.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=2.117056e-12
[layers.0.feed_forward.w1.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=4.166261e-13
[layers.0.feed_forward.w2.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=7.227826e-13
[layers.0.feed_forward.w3.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=4.086736e-13
[layers.0.attention_norm.weight] equal=True, max_abs_diff=0.000000e+00, mean_abs_diff=0.000000e+00
[layers.0.ffn_norm.weight] equal=True, max_abs_diff=0.000000e+00, mean_abs_diff=0.000000e+00
[layers.1.attention.wq.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=1.103546e-13
[layers.1.attention.wk.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=2.372120e-13
[layers.1.attention.wv.weight] equal=True, max_abs_diff=1.862645e-09, mean_abs_diff=3.826575e-12
[layers.1.attention.wo.weight] equal=True, max_abs_diff=1.862645e-09, mean_abs_diff=2.152586e-12
[layers.1.feed_forward.w1.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=3.736954e-13
[layers.1.feed_forward.w2.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=6.429397e-13
[layers.1.feed_forward.w3.weight] equal=True, max_abs_diff=9.313226e-10, mean_abs_diff=3.688944e-13
[layers.1.attention_norm.weight] equal=True, max_abs_diff=0.000000e+00, mean_abs_diff=0.000000e+00
[layers.1.ffn_norm.weight] equal=True, max_abs_diff=0.000000e+00, mean_abs_diff=0.000000e+00
[norm.weight] equal=True, max_abs_diff=0.000000e+00, mean_abs_diff=0.000000e+00
[output.weight] equal=True, max_abs_diff=1.862645e-09, mean_abs_diff=3.105705e-13
```

## References


### FSDP

https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/FSDP2

https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html


### TP
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

https://github.com/pytorch/examples/tree/main/distributed/tensor_parallelism
