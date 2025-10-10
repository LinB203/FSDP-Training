import sys
import os
from pathlib import Path

here = Path(__file__).resolve()
project_root = here.parents[2]
sys.path.insert(0, str(project_root))
from evaluate.configuration_eval import EvalConfig
import json
import torch
import random
import subprocess
import numpy as np
import torch.distributed as dist
import pandas as pd
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist

# hf libs
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from accelerate import init_empty_weights
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage
from diffusers import QwenImageEditPlusPipeline

# Import your model definition here
from models.transformer import (
    Transformer2DModel,
    ModelArgs,
)


# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_models(args, device):

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.scheduler_name_or_path
    )

    vae = (
        AutoencoderKLQwenImage.from_pretrained(args.vae_name_or_path)
        .eval()
        .to(device=device, dtype=torch.bfloat16)
    )
    vae = torch.compile(vae)

    condition_encoder = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.condition_encoder_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to(device=device)
    )
    # condition_encoder = torch.compile(condition_encoder)

    processor = Qwen2VLProcessor.from_pretrained(args.condition_processor_name_or_path)

    # Load main model
    simple_model_config = ModelArgs()
    model = Transformer2DModel.from_model_args(simple_model_config)
    model.load_state_dict(torch.load(args.model_name_or_path, map_location="cpu"))
    model.to(device, dtype=torch.bfloat16)
    # model = torch.compile(model)

    # Load pipeline
    pipe = QwenImageEditPlusPipeline(
        transformer=model,
        vae=vae,
        text_encoder=condition_encoder,
        scheduler=scheduler,
        processor=processor,
        tokenizer=None,
    )

    return {
        "pipe": pipe,
        "device": device,
    }


def init_gpu_env(args):
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    args.local_rank = local_rank
    args.global_rank = global_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=global_rank
    )
    print(
        f"global_rank: {global_rank}, local_rank: {local_rank}, world_size: {world_size}"
    )
    return args


def run_model_and_return_samples(args, state, text, image1=None, image2=None):

    image = Image.open(image1).convert("RGB")
    ori_w, ori_h = image.size
    inputs = {
        "image": image,
        "prompt": text,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": args.guidance_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": args.num_images_per_prompt,
    }

    with torch.no_grad():
        img = state["pipe"](**inputs).images
    # return [i.resize((ori_w, ori_h)) for i in img]
    return img


def main(args):

    args = init_gpu_env(args)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.global_rank, device_specific=False)
    device = torch.cuda.current_device()
    state = initialize_models(args, device)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation prompts
    with open(args.gedit_prompt_path, "r") as f:
        data = json.load(f)

    inference_list = []

    for key, value in tqdm(data.items()):
        outpath = args.output_dir
        os.makedirs(outpath, exist_ok=True)

        prompt = value["prompt"]
        image_path = value["id"]
        inference_list.append([prompt, outpath, key, image_path])

    inference_list = inference_list[args.global_rank :: args.world_size]

    for prompt, output_path, key, image_path in tqdm(inference_list):

        output_path = os.path.join(output_path, image_path)
        real_image_path = os.path.join(args.gedit_image_dir, image_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            continue
        image = run_model_and_return_samples(
            args, state, prompt, image1=real_image_path, image2=None
        )
        image = image[0]
        image.save(output_path)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--model_name_or_path", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default=None, required=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(EvalConfig)
    conf = OmegaConf.merge(schema, config)
    if args.model_name_or_path is not None:
        assert args.output_dir is not None
        conf.model_name_or_path = args.model_name_or_path
        conf.output_dir = args.output_dir
    main(conf)
