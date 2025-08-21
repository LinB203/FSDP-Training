
export TOKENIZERS_PARALLELISM=true

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none


MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))

# torchrun \
#   --nproc_per_node=8 \
#   --nnodes=${WORLD_SIZE} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   fsdp.py \
#   --batch_size 1 \
#   --epochs 3 \
#   --seq_len 1024 \
#   --lr 1e-4

# torchrun \
#   --nproc_per_node=8 \
#   --nnodes=${WORLD_SIZE} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   fsdp.py \
#   --batch_size 1 \
#   --epochs 3 \
#   --seq_len 1024 \
#   --lr 1e-4


torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tp_fsdp.py \
  --batch_size 1 \
  --epochs 3 \
  --seq_len 4096 \
  --lr 1e-4