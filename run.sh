
export TOKENIZERS_PARALLELISM=true

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=WARN

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))


torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 8192 \
  --lr 1e-4  --gradient_checkpointing \
  --enable_ema --ema_decay 0.0 --reshard_after_forward



# torchrun \
#   --nproc_per_node=8 \
#   --nnodes=${WORLD_SIZE} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   train_sp_tp_fsdp.py \
#   --batch_size 1 \
#   --epochs 3 \
#   --seq_len 8192 \
#   --lr 1e-4 \
#   --tp_size 2 


# torchrun \
#   --nproc_per_node=8 \
#   --nnodes=${WORLD_SIZE} \/
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   train_fsdp_real.py \
#   --batch_size 1 \
#   --epochs 3 \
#   --seq_len 16384 \
#   --lr 1e-4 \
#   --qwen_model_name /mnt/data/checkpoints/Qwen/Qwen3-0.6B \
#   --data_path /mnt/data/datasets/Salesforc/e/wikitext

# torchrun \
#   --nproc_per_node=8 \
#   --nnodes=${WORLD_SIZE} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   train_fsdp_real.py \
#   --batch_size 1 \
#   --epochs 3 \
#   --seq_len 8192 \
#   --lr 1e-4 \
#   --qwen_model_name /mnt/data/checkpoints/Qwen/Qwen3-0.6B \
#   --data_path /mnt/data/datasets/Salesforce/wikitext