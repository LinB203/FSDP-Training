
The original code is from [ImgEdit](https://huggingface.co/datasets/sysuyy/ImgEdit).

## Requirements and Installation
Install the required dependencies using `pip`:

```
pip install tqdm tenacity 
pip install -U openai
```



## Eval

### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/imgedit'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  imgedit.yaml \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
  


#!/bin/bash
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

# MODEL_PATH='/mnt/data/checkpoints/Qwen/Qwen-Image-Edit/transformer/merged.pt'
# OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/imgedit/qwen_image_edit'

for step in 1000 6000 11000 16000 21000 26000 31000; do
    for model_type in model_state_dict ema_model; do
         MODEL_PATH="/mnt/data/lb/FSDP-Training/checkpoints/all_data/checkpoints-${step}/${model_type}.pt"
         OUTPUT_DIR="/mnt/data/lb/FSDP-Training/eval_output/imgedit/${step}_${model_type}"
         torchrun \
         --nproc_per_node=8 \
         --nnodes=${WORLD_SIZE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         -m step1_gen_samples \
         imgedit.yaml \
         --model_name_or_path ${MODEL_PATH} \
         --output_dir ${OUTPUT_DIR}
    done
done

MODEL_PATH='/mnt/data/lb/FSDP-Training/checkpoints/checkpoints-6500/model_state_dict.pt'
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/imgedit/text_edit_data_test'
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m step1_gen_samples \
  imgedit.yaml \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation

The benchmark images can be downloaded from huggingface [Benchmark.tar](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar)

```bash
ASSET_ROOT="imgedit_asset"
mkdir -p "$ASSET_ROOT"
wget -O "$ASSET_ROOT/Benchmark.tar" "https://huggingface.co/datasets/sysuyy/ImgEdit/resolve/main/Benchmark.tar"
cd $ASSET_ROOT
tar -xf "Benchmark.tar"
cd ..
```


```bash
# switch to univa env
IMAGE_DIR=${OUTPUT_DIR}
python step2_basic_bench.py \
   --result_img_folder ${IMAGE_DIR} \
   --result_json ${IMAGE_DIR}/imgedit_bench.json \
   --edit_json eval_prompts/basic_edit.json \
   --prompts_json eval_prompts/prompts.json \
   --origin_img_root ${ASSET_ROOT}/Benchmark/singleturn \
   --api_key ${OPENAI_API_KEY} 


cd /mnt/data/lb/FSDP-Training/evaluate/imgedit
ASSET_ROOT="/mnt/data/lb/Remake/FlowWorld/univa/eval/imgedit/imgedit_asset"
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/imgedit/qwen_image_edit'
# OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/imgedit/text_edit_data_test'
IMAGE_DIR=${OUTPUT_DIR}
python step2_basic_bench.py \
   --result_img_folder ${IMAGE_DIR} \
   --result_json ${IMAGE_DIR}/imgedit_bench.json \
   --edit_json eval_prompts/basic_edit.json \
   --prompts_json eval_prompts/prompts.json \
   --origin_img_root ${ASSET_ROOT}/Benchmark/singleturn \
   --api_key "sk-mDPOkWlsXRECQph714C2E845AaAa4011A1Ca8b7875048970" \
   --base_url "https://api.bltcy.ai/v1"



```

### Summary  

```bash
python step3_get_avgscore.py \
   --input ${IMAGE_DIR}/imgedit_bench.json \
   --meta_json eval_prompts/basic_edit.json \
   --output_json ${IMAGE_DIR}.json
cat ${IMAGE_DIR}.json
```