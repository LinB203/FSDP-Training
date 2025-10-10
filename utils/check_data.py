import gradio as gr
import json
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

import transformers
import copy
import torch
import concurrent.futures
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def pil_center_crop_resize(pil_img, size):

    if not isinstance(pil_img, Image.Image):
        raise TypeError("pil_img must be a PIL.Image")

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive (width, height)")

    old_w, old_h = pil_img.size  # PIL: size -> (width, height)
    old_aspect = old_w / old_h
    new_aspect = width / height

    # --- center crop to match target aspect ratio ---
    if abs(old_aspect - new_aspect) < 1e-9:
        cropped = pil_img
    elif old_aspect > new_aspect:
        # too wide -> crop left/right
        new_w = int(round(old_h * new_aspect))
        left = (old_w - new_w) // 2
        right = left + new_w
        cropped = pil_img.crop((left, 0, right, old_h))
    else:
        # too tall -> crop top/bottom
        new_h = int(round(old_w / new_aspect))
        upper = (old_h - new_h) // 2
        lower = upper + new_h
        cropped = pil_img.crop((0, upper, old_w, lower))
    return cropped.resize((width, height))


def FluxKontextImageScale(image):
    width, height = image.size
    aspect_ratio = width / height
    _, width, height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )

    return pil_center_crop_resize(image, (width, height))


# —— Gradio 相关函数 ——
data = []
img_root = ""
edit_img_root = ""


def load_json(json_path, image_root, edit_image_root):
    global data, img_root, edit_img_root
    img_root = image_root.strip()
    edit_img_root = edit_image_root.strip()
    try:
        with open(json_path.strip(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return f"成功加载 {len(data)} 条原始数据。"
    except Exception as e:
        return f"加载 JSON 文件出错：{e}"


def show_random_sample():
    global data
    if len(data) == 0:
        return "请输入 JSON 文件路径 并点击加载"
    if len(img_root) == 0:
        return "请输入 图片根目录 并点击加载"
    if len(edit_img_root) == 0:
        return "请输入 编辑图片根目录 并点击加载"
    sample = random.choice(data)
    img_f = sample.get("image", [])
    fulls = [os.path.join(img_root, img_f[0]), os.path.join(edit_img_root, img_f[1])]
    fulls = [Image.open(i) for i in fulls]

    if "TextAtlas5M" in img_root:
        fulls[0] = FluxKontextImageScale(fulls[0])
    else:
        fulls[0] = fulls[0].resize(fulls[1].size, resample=Image.BICUBIC)
    assert fulls[0].size == fulls[1].size
    text = sample["text"]
    return fulls, text


# —— Gradio 界面搭建 ——
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 数据集辅助验证工具")

    with gr.Row():
        json_path = gr.Textbox(label="JSON 文件路径")
        image_root = gr.Textbox(label="图片根目录")
        edit_img_root = gr.Textbox(label="编辑图片根目录")
        load_btn = gr.Button("加载 JSON（点我）")
    load_status = gr.Textbox(label="加载状态", interactive=False)

    gallery = gr.Gallery(label="图像预览", columns=4)
    text_box = gr.Textbox(label="对话内容", lines=2, interactive=False)
    random_btn = gr.Button("随机查看样本（点我）")

    # 事件绑定
    load_btn.click(
        load_json, inputs=[json_path, image_root, edit_img_root], outputs=load_status
    )
    random_btn.click(show_random_sample, outputs=[gallery, text_box])

# server_port = 7888
demo.launch(
    # server_port=server_port,
    allowed_paths=["/mnt"]
)
