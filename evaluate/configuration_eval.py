from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EvalConfig:
    model_name_or_path: str
    condition_encoder_name_or_path: str
    condition_processor_name_or_path: str
    vae_name_or_path: str
    scheduler_name_or_path: str

    seed: int = 42
    allow_tf32: bool = False

    output_dir: str = "./output"

    num_images_per_prompt: int = 1
    num_inference_steps: int = 32
    guidance_scale: float = 3.5  # Used in Flux
    num_samples_per_prompt: int = 1

    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    # imgedit
    imgedit_prompt_path: str = "univa/eval/imgedit/basic_edit.json"
    imgedit_image_dir: str = "/mnt/data/lb/Remake/imgedit_bench_eval_images"

    # gedit
    gedit_prompt_path: str = "univa/eval/gedit/gedit_edit.json"
    gedit_image_dir: str = "/mnt/data/lb/Remake/gedit_bench_eval_images"
