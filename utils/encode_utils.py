import math
import torch
from typing import List, Dict
from transformers import Qwen2_5_VLForConditionalGeneration


def get_qwen_prompt_embeds(
    condition_encoder: Qwen2_5_VLForConditionalGeneration,
    model_inputs: Dict,
    device: torch.device,
    dtype: torch.dtype,
    prompt_template_encode_start_idx: int = 64,
):
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    drop_idx = prompt_template_encode_start_idx

    outputs = condition_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(
        hidden_states, model_inputs.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [
        torch.ones(e.size(0), dtype=torch.long, device=e.device)
        for e in split_hidden_states
    ]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ]
    )
    prompt_embeds_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, prompt_embeds_mask


def vae_encode_and_pack(vae, image, latents_mean, latents_std, device):
    def _batch_encode(vae, x):
        with torch.no_grad():
            model_input = vae.encode(x).latent_dist.sample().squeeze(2)  # squeeze t
        model_input = (model_input - latents_mean) / latents_std
        model_input = pack_latents(model_input)
        return model_input

    if all(i.shape == image[0].shape for i in image):
        image = torch.stack(image).to(device, vae.dtype, non_blocking=True)
        latent = _batch_encode(vae, image)
    else:
        latent = [
            _batch_encode(
                vae, img.unsqueeze(0).to(device, vae.dtype, non_blocking=True)
            )[0]
            for img in image
        ]
    return latent


def add_noise(latent, scheduler):
    def _batch_add_noise(x, scheduler):
        noise = torch.randn_like(x)
        bsz = x.shape[0]
        sigmas = torch.sigmoid(
            1.0 * torch.randn((bsz,), device=x.device, dtype=torch.float32)
        )
        sigmas = apply_shift(scheduler, sigmas, noise)
        timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
        while sigmas.ndim < x.ndim:
            sigmas = sigmas.unsqueeze(-1)
        noisy_model_input = (1.0 - sigmas) * x + sigmas * noise
        return noise, noisy_model_input, timesteps

    if all(i.shape == latent[0].shape for i in latent):
        latent = torch.stack(latent) if isinstance(latent, list) else latent
        noise, noisy_model_input, timesteps = _batch_add_noise(latent, scheduler)
    else:
        noise, noisy_model_input, timesteps = [], [], []
        for i in latent:
            noise_i, noisy_model_input_i, timesteps_i = _batch_add_noise(
                i.unsqueeze(0), scheduler
            )
            noise.append(noise_i[0])
            noisy_model_input.append(noisy_model_input_i[0])
            timesteps.append(timesteps_i[0])
    return noise, noisy_model_input, timesteps


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def apply_shift(scheduler, sigmas, noise):
    # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
    # Resolution-dependent shift value calculation used by official Flux inference implementation
    image_seq_len = noise.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    shift = math.exp(mu)
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas


def pack_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents
