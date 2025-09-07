import os
import random
import string

from tqdm import tqdm
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from diffusers.image_processor import VaeImageProcessor

from utils.dataset_utils import PREFERRED_KONTEXT_RESOLUTIONS, calculate_dimensions


class RandomDataset(Dataset):

    def __init__(
        self,
        args,
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
        length = random.randint(10, 512)
        return " ".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def __getitem__(self, idx):
        img_ref = self.get_random_pil_image()
        img_gt = deepcopy(img_ref)
        w, h = img_ref.size
        aspect_ratio = w / h
        _, w, h = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        # w, h = 128, 128
        img_ref = self.image_processor.resize(img_ref, w, h)
        prompt_image = img_ref  # PIL.Image

        # 3 1 h w, [-1, 1]
        img_ref = self.image_processor.preprocess(img_ref, w, h).transpose(0, 1)
        img_gt = self.image_processor.preprocess(img_gt, w, h).transpose(0, 1)

        prompt = self.get_random_prompt()
        template = self.prompt_template_encode
        txt = template.format(prompt)
        return dict(txt=txt, prompt_image=prompt_image, img_gt=img_gt, img_ref=img_ref)
