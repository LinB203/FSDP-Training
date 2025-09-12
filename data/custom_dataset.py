import os
import sys
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from diffusers.image_processor import VaeImageProcessor

from pathlib import Path

here = Path(__file__).resolve()
project_root = here.parents[1]
sys.path.insert(0, str(project_root))
from utils.dataset_utils import PREFERRED_KONTEXT_RESOLUTIONS, calculate_dimensions

text_edit_temp = [
    'change "{}" to "{}"',
    'replace "{}" to "{}"',
    'replace "{}" with "{}"',
    'replace the text "{}" with "{}"',
    'modify "{}" to "{}"',
    'Help me change "{}" to "{}"',
    'Can you remove "{}", add "{}"',
    'I want to change the letter "{}" to "{}"',
    '将文本 "{}" 替换为 "{}"',
    '更改文本 "{}" 为 "{}"',
    '把 "{}" 改成 "{}"',
    '修改 "{}" 为 "{}"',
    '移除 "{}" 为 "{}"',
    '替换 "{}" 为 "{}"',
]
text_edit_temp_wbox = [
    'change "{}" to "{}" within red box',
    'replace "{}" to "{}" within red box',
    'replace "{}" with "{}" within red box',
    'replace the text "{}" with "{}" within red box',
    'modify "{}" to "{}" within red box',
    'Help me change "{}" to "{}" within red box',
    'Can you remove "{}", add "{}" within red box',
    'I want to change the letter "{}" to "{}" within red box',
    '将红框里的文本 "{}" 替换为 "{}"',
    '更改红框内的文本 "{}" 为 "{}"',
    '把红框里的 "{}" 改成 "{}"',
    '修改红框内的 "{}" 为 "{}"',
    '移除红框里的 "{}" 为 "{}"',
    '替换红框内的 "{}" 为 "{}"',
]


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    assert pil_img.mode == "RGB"
    arr = np.asarray(pil_img)  # shape = (H, W, 3), dtype = uint8 (通常)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    assert cv2_img.ndim == 3 and cv2_img.shape[2] == 3
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# 画红色 polygon 框（向外延伸）
def expand_polygon(pts, height, width, scale_max=1.2):
    scale = random.uniform(1.0, scale_max)
    center = np.mean(pts, axis=0)
    expanded = (pts - center) * scale + center
    # 限制边界在图像尺寸内
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded


def draw_red_box(image: Image.Image, rec_polys):

    image = pil_to_cv2(image)

    height, width = image.shape[:2]
    alpha = random.uniform(0.5, 1.0)
    default_box_width_at_1024 = 30
    thickness = random.randint(
        4, int(default_box_width_at_1024 * min(height, width) / 1024)
    )

    overlay = image.copy()
    pts = np.array([[int(x), int(y)] for x, y in rec_polys], dtype=np.float32)
    expanded_pts = expand_polygon(pts, height, width)

    cv2.polylines(
        overlay,
        [expanded_pts.astype(np.int32)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=thickness,
    )
    # 将overlay混合回原图
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return cv2_to_pil(image)


class CustomDataset(Dataset):

    def __init__(
        self,
        args,
        image_processor: VaeImageProcessor,
    ):
        data_txt = args.dataset_config.data_txt
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]
        self.reversed_text_edit_ratio = args.dataset_config.reversed_text_edit_ratio
        self.box_guidance_text_edit_ratio = (
            args.dataset_config.box_guidance_text_edit_ratio
        )
        self.data = []
        self._load_data()

        self.image_processor = image_processor
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        for dataset in self.datasets:
            src_image_root, dst_image_root, json_file = dataset.split(",")

            # Load json file
            with open(json_file, "r") as f:
                data = json.load(f)
            dataset_data = []
            for line in tqdm(data):
                assert "image" in line
                # Ensure `image` is a list
                image = line["image"]
                assert isinstance(image, list) and len(image) == 2
                # Convert image path to absolute path
                image[0] = os.path.join(src_image_root, image[0])
                image[1] = os.path.join(dst_image_root, image[1])
                line["img_ref"] = image[0]
                line["img_gt"] = image[1]
                dataset_data.append(line)

            print(f"Load {len(dataset_data)} data from {json_file}.")
            self.data.extend(dataset_data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            return self.getitem(item)
        except Exception as e:
            print(f"Error with {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def getitem(self, item):
        img_ref = Image.open(item["img_ref"]).convert("RGB")
        img_gt = Image.open(item["img_gt"]).convert("RGB")
        w, h = img_ref.size
        aspect_ratio = w / h
        _, w, h = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        if "rec_polys" in item:  # for text-edit
            ref_text = item["text"][0]
            gt_text = item["text"][1]
            prompt = random.choice(text_edit_temp).format(ref_text, gt_text)
            if self.reversed_text_edit_ratio > random.random():
                tmp = deepcopy(img_ref)
                img_ref = img_gt
                img_gt = tmp
                prompt = random.choice(text_edit_temp).format(gt_text, ref_text)
            elif self.box_guidance_text_edit_ratio > random.random():
                img_ref = draw_red_box(img_ref, item["rec_polys"])
                prompt = (
                    random.choice(text_edit_temp).format(ref_text, gt_text)
                    if random.random() > 0.5
                    else random.choice(text_edit_temp_wbox).format(ref_text, gt_text)
                )
        elif "aug_text" in item:  # for NHR-Edit
            if random.random() > 0.34:
                prompt = random.choice(item["aug_text"])
            else:
                prompt = item["text"]
        else:
            prompt = item["text"]

        # w, h = 128, 128
        img_ref = self.image_processor.resize(img_ref, h, w)
        prompt_image = img_ref  # PIL.Image

        # 3 1 h w, [-1, 1]
        img_ref = self.image_processor.preprocess(img_ref, h, w).transpose(0, 1)
        img_gt = self.image_processor.preprocess(img_gt, h, w).transpose(0, 1)

        template = self.prompt_template_encode
        txt = template.format(prompt)
        return dict(txt=txt, prompt_image=prompt_image, img_gt=img_gt, img_ref=img_ref)


if __name__ == "__main__":
    from diffusers.image_processor import VaeImageProcessor
    from torch.utils.data import DataLoader

    # If they are tensors, optionally convert one back to PIL and save for quick visual check
    def tensor_to_pil(tensor_or_arr, save_path=None):
        arr = tensor_or_arr.detach().cpu().numpy()
        print(arr.shape, arr.max(), arr.min())
        arr = arr[:, 0]  # -> (C,H,W)
        # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
        # arr expected in [-1,1] or [0,1]
        arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if save_path:
            pil.save(save_path)
        return pil

    class SimpleNamespace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    dataset_config = SimpleNamespace(
        data_txt="/mnt/data/lb/FSDP-Training/configs/test.txt",
        reversed_text_edit_ratio=0.5,
        box_guidance_text_edit_ratio=0.5,
    )
    args = SimpleNamespace(dataset_config=dataset_config)
    image_processor = VaeImageProcessor(vae_scale_factor=8 * 2)
    ds = CustomDataset(args, image_processor)
    dataloader = DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x
    )
    for i, item in enumerate(tqdm(dataloader)):
        print(item)
        item = item[0]
        img_ref = item.get("img_ref", None)
        img_gt = item.get("img_gt", None)

        print(item["txt"])
        pil_ref = tensor_to_pil(img_ref, save_path=f"sample_{i}_ref.png")
        pil_gt = tensor_to_pil(img_gt, save_path=f"sample_{i}_gt.png")
        if pil_ref:
            print("Saved img_ref as PNG.")
        if pil_gt:
            print("Saved img_gt as PNG.")
        if i == 3:
            break
