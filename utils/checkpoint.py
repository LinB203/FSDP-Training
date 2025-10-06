import os
import time
import shutil
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor


MODEL_CHECKPOINT = "model_state_dict.pt"
EMA_MODEL_CHECKPOINT = "ema_model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
PARAMS = "params"


class Checkpointer:
    def __init__(
        self,
        folder: str,
        dcp_api: bool,
        model: torch.distributed.fsdp.FSDPModule,
        checkpoints_total_limit: int = None,
        enable_ema: bool = True,
        decay: float = 0.99,
        fsdp_resharded: bool = False,
    ):
        self.folder = folder
        self.dcp_api = dcp_api
        self.checkpoints_total_limit = checkpoints_total_limit

        self.decay = decay
        self.fsdp_resharded = fsdp_resharded
        self.shadow = {}
        self.backup = {}
        self.ema_is_registered = False
        if enable_ema:
            self.model = model

    def load_model(self, model: FSDPModule, last_model_checkpoint_folder: str):
        full_sd = torch.load(
            f"{last_model_checkpoint_folder}/{MODEL_CHECKPOINT}",
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        if self.dcp_api:
            set_model_state_dict(
                model=model,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        # choose `assign=True` since we cannot call `copy_` on meta tensor
        model.load_state_dict(sharded_sd, strict=False, assign=True)

    def load_optim(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
        last_model_checkpoint_folder: str,
    ):
        is_rank_zero = torch.distributed.get_rank() == 0
        full_sd = torch.load(
            f"{last_model_checkpoint_folder}/{OPTIM_CHECKPOINT}",
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        if self.dcp_api:
            set_optimizer_state_dict(
                model=model,
                optimizers=opt,
                optim_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        _init_optim_state(opt)
        param_groups = opt.state_dict()["param_groups"]
        state = opt.state_dict()["state"]

        full_param_groups = full_sd["param_groups"]
        full_state = full_sd["state"]

        for param_group, full_param_group in zip(param_groups, full_param_groups):
            for key, value in full_param_group.items():
                if key == PARAMS:
                    continue
                param_group[key] = value
            for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
                if pid not in state:
                    continue
                param_state = state[pid]
                full_param_state = full_state.get(full_pid)
                if full_param_state is None:
                    if is_rank_zero:
                        print(f"WARN: param [{full_pid}] does NOT have param_state")
                    continue
                for attr, full_tensor in full_param_state.items():
                    sharded_tensor = param_state[attr]
                    if isinstance(sharded_tensor, DTensor):
                        # exp_avg is DTensor
                        param_state[attr] = distribute_tensor(
                            full_tensor,
                            sharded_tensor.device_mesh,
                            sharded_tensor.placements,
                        )
                    else:
                        # step is plain tensor
                        param_state[attr] = full_tensor
        opt.load_state_dict(
            {
                "param_groups": param_groups,
                "state": state,
            }
        )

    def _get_full_model_state_dict(self, model: FSDPModule):
        if self.dcp_api:
            return get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        return cpu_state_dict

    def _get_full_optimizer_state_dict(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
        sharded_sd = opt.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def save(
        self,
        model: FSDPModule,
        optim: torch.optim.Optimizer,
        basename: str,
    ):
        is_rank_zero = torch.distributed.get_rank() == 0
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        if is_rank_zero:
            self.clean_outdate_checkpoints()
            new_checkpoint_folder = f"{self.folder}/{basename}"
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)
        torch.distributed.barrier()
        cpu_state_dict = {}
        for param_name, sharded_param in self.shadow.items():
            full_param = sharded_param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        if torch.distributed.get_rank() == 0:
            new_ema_model_checkpoint = f"{new_checkpoint_folder}/{EMA_MODEL_CHECKPOINT}"
            torch.save(cpu_state_dict, new_ema_model_checkpoint)
        torch.distributed.barrier()

    def get_resume_path(self, resume_from_checkpoint):
        resume_checkpoint_path = None
        initial_global_step = 0
        is_rank_zero = torch.distributed.get_rank() == 0
        if resume_from_checkpoint:
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                if os.path.exists(self.folder):
                    dirs = os.listdir(self.folder)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None
                else:
                    path = None

            if path is None:
                if is_rank_zero:
                    print(
                        f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                    )
                resume_checkpoint_path = None
                initial_global_step = 0
            else:
                if is_rank_zero:
                    print(f"Resuming from checkpoint {path}")
                resume_checkpoint_path = os.path.join(self.folder, path)
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
        return resume_checkpoint_path, initial_global_step

    def clean_outdate_checkpoints(
        self,
    ):
        if self.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.folder)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= self.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                print(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(
                        self.folder,
                        removing_checkpoint,
                    )
                    shutil.rmtree(removing_checkpoint)

    def ema_register(self):
        if self.ema_is_registered:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert isinstance(param, DTensor), f"{name}"
                self.shadow[name] = param.data.clone().float()
        self.ema_is_registered = True

    def ema_update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                assert isinstance(param, DTensor), f"{name}"
                assert isinstance(self.shadow[name], DTensor), f"{name}"
                new_average = (
                    1.0 - self.decay
                ) * param.data.float() + self.decay * self.shadow[name].float()
                self.shadow[name] = new_average.clone().float()

    def ema_apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                shadow_tensor = self.shadow[name]
                self.backup[name] = param.data.clone()
                assert isinstance(shadow_tensor, DTensor), f"{name}"
                assert isinstance(param.data, DTensor), f"{name}"
                param.data.copy_(shadow_tensor.to(param.data.device))

    def ema_restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                assert isinstance(self.backup[name], DTensor), f"{name}"
                assert isinstance(param.data, DTensor), f"{name}"
                param.data.copy_(self.backup[name])
