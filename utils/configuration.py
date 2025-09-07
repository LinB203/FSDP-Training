from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    seed: int = 42
    mixed_bf16_precision: bool = True
    gradient_checkpointing: bool = False
    dcp_api: bool = False
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    max_grad_norm: Optional[float] = 1.0
    log_interval: int = 1
    validation_steps: int = 500
    save_interval: int = 1000
    wandb_proj_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    output_dir: str = "checkpoints"
    weight_init: Optional[str] = None
    resume_from_checkpoint: Optional[str] = "latest"
    checkpoints_total_limit: Optional[int] = None
    num_to_explicit_prefetching: Optional[int] = None


@dataclass
class DatasetConfig:
    data_txt: Optional[str] = None
    val_data_txt: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4

    reversed_text_edit_ratio: float = 0.0
    box_guidance_text_edit_ratio: float = 0.0


@dataclass
class ModelConfig:
    condition_encoder_name_or_path: str
    condition_processor_name_or_path: str
    vae_name_or_path: str
    scheduler_name_or_path: str


@dataclass
class Config:
    training_config: TrainingConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
