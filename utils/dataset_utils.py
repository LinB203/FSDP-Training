import math


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


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None
