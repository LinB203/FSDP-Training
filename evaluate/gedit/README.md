
The original code is from [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md).

## Requirements and Installation

```
pip install megfile openai 
```

## Prepare Source Images
Prepare the original image and metadata json following the example code in `step0_generate_image_example.py`

```bash
GEDIT_ASSET="/path/to/gedit_asset"
python step0_prepare_gedit.py --save_path ${GEDIT_ASSET} --json_file_path gedit_edit.json
```

The file directory structure of the original image：
```folder
${GEDIT_ASSET}/
│   └── fullset/
│       └── edit_task/
│           ├── cn/  # Chinese instructions
│           │   ├── key1.png
│           │   ├── key2.png
│           │   └── ...
│           └── en/  # English instructions
│               ├── key1.png
│               ├── key2.png
│               └── ...
```

## Eval


### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/gedit'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  gedit.yaml \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}


  
# switch to univa env

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
# OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/qwen_image_edit'
MODEL_PATH='/mnt/data/lb/FSDP-Training/checkpoints/checkpoints-6500/model_state_dict.pt'
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/text_edit_data_test'
torchrun \
  --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m step1_gen_samples \
  gedit.yaml \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation

Write your gpt-api-key to `secret_t2.env` and gpt-api-url to `secret_t2_url.env`.

```bash
IMAGE_DIR=${OUTPUT_DIR}
python step2_gedit_bench.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --source_path ${GEDIT_ASSET}
    
GEDIT_ASSET="/mnt/data/lb/Remake/FlowWorld/univa/eval/gedit/gedit_asset"
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/text_edit_data_test'
IMAGE_DIR=${OUTPUT_DIR}
python step2_gedit_bench.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --source_path ${GEDIT_ASSET}


GEDIT_ASSET="/mnt/data/lb/Remake/FlowWorld/univa/eval/gedit/gedit_asset"
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/qwen_image_edit'
IMAGE_DIR=${OUTPUT_DIR}
python step2_gedit_bench.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --source_path ${GEDIT_ASSET}
  
```

### Summary
```bash
OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/qwen_image_edit'
IMAGE_DIR=${OUTPUT_DIR}
python step3_calculate_statistics.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --language en > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt

OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/text_edit_data_test'
IMAGE_DIR=${OUTPUT_DIR}
python step3_calculate_statistics.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --language en > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt








OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/qwen_image_edit'
IMAGE_DIR=${OUTPUT_DIR}
python step3_calculate_statistics.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --language cn > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt

OUTPUT_DIR='/mnt/data/lb/FSDP-Training/eval_output/gedit/text_edit_data_test'
IMAGE_DIR=${OUTPUT_DIR}
python step3_calculate_statistics.py \
    --model_name qwen_image_edit \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --language cn > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt
```
