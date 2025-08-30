import os
import wandb
import random
import string
import argparse

from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from typing import List, Dict
from omegaconf import OmegaConf

import torch
from torch import nn
from torch import distributed as dist
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler

# hf libs
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoConfig,
    Qwen2VLProcessor,
)
from accelerate import init_empty_weights
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Import your model definition here
from models.transformer import (
    Transformer2DModel,
    ModelArgs,
    TransformerBlock,
)
from utils.configuration import Config
from utils.checkpoint import Checkpointer
from utils.merge_safetensors import load_safetensors
from utils.fsdp2_warpper import FSDP2_mix_warpper, FSDP2_warpper
from utils.log_utils import rank_log, get_logger, get_memory_allocated
from utils.dataset_utils import PREFERRED_KONTEXT_RESOLUTIONS, calculate_dimensions
from utils.dist_utils import setup_distributed_env, cleanup_distributed_env, set_seed
from utils.encode_utils import get_qwen_prompt_embeds, vae_encode_and_pack, add_noise


class RandomDataset(Dataset):
    """Holds fixed-length token blocks as torch tensors."""

    def __init__(
        self,
        image_processor: VaeImageProcessor,
    ):
        self.image_processor = image_processor
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def __len__(self):
        return 1000000

    def get_random_pil_image(self):
        h = random.randint(512, 2048)
        w = random.randint(512, 2048)
        img_tensor = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
        img = Image.fromarray(img_tensor.numpy())
        return img

    def get_random_prompt(self):
        length = random.randint(10, 64)
        return " ".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def __getitem__(self, idx):
        img_ref = self.get_random_pil_image()
        img_gt = deepcopy(img_ref)
        w, h = img_ref.size
        aspect_ratio = w / h
        # _, w, h = min(
        #     (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        # )
        w, h = 128, 128
        img_ref = self.image_processor.resize(img_ref, w, h)
        prompt_image = img_ref  # PIL.Image

        # 3 1 h w, [-1, 1]
        img_ref = self.image_processor.preprocess(img_ref, w, h).transpose(0, 1)
        img_gt = self.image_processor.preprocess(img_gt, w, h).transpose(0, 1)

        prompt = self.get_random_prompt()
        template = self.prompt_template_encode
        txt = template.format(prompt)
        return dict(txt=txt, prompt_image=prompt_image, img_gt=img_gt, img_ref=img_ref)


class DataCollator:
    def __init__(
        self,
        processor: Qwen2VLProcessor,
    ):
        self.processor = processor

    def __call__(self, instances: List[Dict]) -> Dict:
        # List[str]
        txt = [i["txt"] for i in instances]
        # List[PIL.Image]
        prompt_image = [i["prompt_image"] for i in instances]
        # List[Tensor], Tensor 1 3 h w, [-1, 1]
        img_ref = [i["img_ref"] for i in instances]
        # List[Tensor], Tensor 1 3 h w, [-1, 1]
        img_gt = [i["img_gt"] for i in instances]
        # prompt_inputs: input_ids (b, l), attention_mask (b, l), pixel_values (l', d), image_grid_thw (b, 3)
        prompt_inputs = self.processor(
            text=txt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        )
        return dict(prompt_inputs=prompt_inputs, img_ref=img_ref, img_gt=img_gt)


def prepare_dataloader(args, image_processor, processor, rank, world_size):
    # create dataset
    dataset = RandomDataset(image_processor)
    collator = DataCollator(processor)

    # Create DistributedSampler when world_size > 1
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.dataset_config.batch_size,
        sampler=None,
        shuffle=True if sampler is None else False,
        num_workers=args.dataset_config.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    return dataloader, sampler


def train(args):
    logger = get_logger()
    setup_distributed_env()
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.training_config.mixed_bf16_precision:
        weight_dtype = torch.bfloat16
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if global_rank == 0:
        print(f"World size: {world_size}, local_rank: {local_rank}, device: {device}")
        if args.training_config.wandb_proj_name is not None:
            wandb.init(
                project=args.training_config.wandb_proj_name,
                name=str(args.training_config.wandb_run_name),
                config=OmegaConf.to_container(args, resolve=True),
            )

    set_seed(
        args.training_config.seed, global_rank, device_specific=False
    )  # we need same seed to init meta weights

    # ---- load tokenizer ----
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_config.condition_processor_name_or_path
    )
    condition_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_config.condition_encoder_name_or_path,
        torch_dtype=weight_dtype,
        attn_implementation="sdpa",
    ).eval()
    FSDP2_warpper(None, condition_encoder, main_block=None, fp32=False)
    torch.cuda.empty_cache()
    if global_rank == 0:
        print(
            f"Fully_shard condition encoder, memory allocated: {get_memory_allocated()} GiB"
        )

    # ---- create model ----
    simple_model_config = ModelArgs()
    with init_empty_weights():
        model = Transformer2DModel.from_model_args(simple_model_config)
    model.gradient_checkpointing = args.training_config.gradient_checkpointing
    model.train()
    FSDP2_mix_warpper(None, model, TransformerBlock, norm_to_fp32=nn.LayerNorm)
    if args.training_config.weight_init is not None:
        load_safetensors(model, args.training_config.weight_init)
    torch.cuda.empty_cache()
    if global_rank == 0:
        print(f"Fully_shard main model, memory allocated: {get_memory_allocated()} GiB")

    # ---- load vae ----
    vae = AutoencoderKLQwenImage.from_pretrained(args.model_config.vae_name_or_path)
    vae = vae.eval().to(device=device, dtype=weight_dtype)
    vae = torch.compile(vae)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1)
        .to(device, weight_dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1)
        .to(device, weight_dtype)
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_config.scheduler_name_or_path
    )
    image_processor = VaeImageProcessor(vae_scale_factor=8 * 2)
    torch.cuda.empty_cache()
    if global_rank == 0:
        print(f"Loading vae, memory allocated: {get_memory_allocated()} GiB")

    # ---- load checkpoint if present ----
    checkpointer = Checkpointer(
        args.training_config.output_dir,
        dcp_api=args.training_config.dcp_api,
        checkpoints_total_limit=args.training_config.checkpoints_total_limit,
    )
    resume_checkpoint_path, initial_global_step = checkpointer.get_resume_path(
        args.training_config.resume_from_checkpoint
    )
    if resume_checkpoint_path is not None:
        checkpointer.load_model(model, resume_checkpoint_path)
    else:
        model.to_empty(device=device)
        model.reset_parameters()

    # ---- prepare optimizer ----
    # if global_rank == 0:
    #     for n, p in model.named_parameters():
    #         print(n, p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.training_config.learning_rate
    )
    if resume_checkpoint_path is not None:
        checkpointer.load_optim(model, optimizer, resume_checkpoint_path)

    set_seed(args.training_config.seed, global_rank, device_specific=True)
    dataloader, _ = prepare_dataloader(
        args, image_processor, processor, global_rank, world_size
    )

    # ---- training loop ----
    max_grad_norm = args.training_config.max_grad_norm
    batch_size = args.dataset_config.batch_size
    grad_accum_steps = args.training_config.grad_accum_steps
    total_batch_size = batch_size * world_size * grad_accum_steps
    if global_rank == 0:
        print("***** Running training *****")
        print(f"  Num batches each epoch = {len(dataloader)}")
        print(f"  Instantaneous batch size per device = {batch_size}")
        print(f"  Gradient Accumulation steps = {grad_accum_steps}")
        print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    running_loss = 0.0
    accum_loss = 0.0
    global_step = initial_global_step
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        prompt_inputs = batch["prompt_inputs"].to(device, non_blocking=True)
        image_ref = batch["img_ref"]
        image_gt = batch["img_gt"]

        # b s 64 or List[Tensor(s 64)]
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(
                condition_encoder,
                prompt_inputs,
                device,
                weight_dtype,
            )
            latent_ref = vae_encode_and_pack(
                vae, image_ref, latents_mean, latents_std, device
            )
            latent_gt = vae_encode_and_pack(
                vae, image_gt, latents_mean, latents_std, device
            )
        noise, latent, timestep = add_noise(latent_gt, scheduler)
        latent_model_input = torch.cat([latent, latent_ref], dim=1)

        img_shapes = [
            [
                (1, i.shape[-2] // 16, i.shape[-1] // 16),
                (1, j.shape[-2] // 16, j.shape[-1] // 16),
            ]
            for i, j in zip(image_gt, image_ref)
        ]
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        with autocast(device_type="cuda", dtype=weight_dtype):
            # forward
            print(
                f"latent_model_input: {latent_model_input.shape}, prompt_embeds: {prompt_embeds.shape}, img_shapes: {img_shapes}, txt_seq_lens: {txt_seq_lens}"
            )
            noise_pred = model(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        noise_pred = noise_pred[:, : latent.size(1)]
        target = noise - latent_gt
        micro_loss = ((noise_pred.float() - target.float()) ** 2).mean()
        accum_loss += micro_loss.item()
        loss_for_backward = micro_loss / grad_accum_steps

        loss_for_backward.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:

            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss_this_step = accum_loss / grad_accum_steps
            running_loss += avg_loss_this_step
            accum_loss = 0.0

            # logging
            if (
                global_rank == 0
                and global_step % args.training_config.log_interval == 0
            ):
                avg = running_loss / args.training_config.log_interval
                wandb.log({"train/loss": avg}, step=global_step)
                print(f"Step[{global_step}]: loss-{avg:.6f}")
                running_loss = 0.0

            # checkpoint
            if (
                global_step != 0
                and global_step % args.training_config.save_interval == 0
            ):
                checkpointer.save(model, optimizer, f"checkpoints-{global_step}")

    if (batch_idx + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    checkpointer.save(model, optimizer, f"checkpoints-{global_step}")

    print("Training complete.")
    cleanup_distributed_env()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(Config)
    conf = OmegaConf.merge(schema, config)
    train(conf)


if __name__ == "__main__":
    main()
