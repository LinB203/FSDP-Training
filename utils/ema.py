import safetensors
import torch
from torch.distributed._tensor import DTensor
import torch.distributed as dist


class EMA:
    def __init__(self, model, decay: float = 0.99, fsdp_resharded: bool = False):
        self.model = model
        self.decay = decay
        self.fsdp_resharded = fsdp_resharded
        self.shadow = {}
        self.backup = {}
        self.is_registered = False

    def register(self):
        if self.is_registered:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert isinstance(param, DTensor), f"{name}"
                self.shadow[name] = param.data.clone().float()
        self.is_registered = True

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                assert isinstance(param, DTensor), f"{name}"
                assert isinstance(self.shadow[name], DTensor), f"{name}"
                new_average = (
                    1.0 - self.decay
                ) * param.data.float() + self.decay * self.shadow[name].float()
                self.shadow[name] = new_average.clone().float()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                shadow_tensor = self.shadow[name]
                self.backup[name] = param.data.clone()
                assert isinstance(shadow_tensor, DTensor), f"{name}"
                assert isinstance(param.data, DTensor), f"{name}"
                param.data.copy_(shadow_tensor.to(param.data.device))

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                assert isinstance(self.backup[name], DTensor), f"{name}"
                assert isinstance(param.data, DTensor), f"{name}"
                param.data.copy_(self.backup[name])

    def state_dict_full_rank0_cpu(self, path):
        cpu_state_dict = {}
        for param_name, sharded_param in self.shadow.items():
            if self.fsdp_resharded:
                full_param = sharded_param.full_tensor()
            else:
                full_param = sharded_param
            cpu_state_dict[param_name] = full_param.cpu()
        if is_main_process():
            torch.save(cpu_state_dict, path)
            print(f"EMA weights saved to {path}")
        torch.distributed.barrier()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0
