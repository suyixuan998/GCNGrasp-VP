from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
import supervision as sv

CLIENT_KWARGS = {
    # https://help.aliyun.com/model-studio/getting-started/models
    "bailian": dict(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),

    "xiaomi": dict(
        api_key=os.getenv("XIAOAI_API_KEY"),
        base_url="https://xiaoai.plus/v1"
    )
}


def process_image(img: np.ndarray,
                  max_img_size: int) -> str:
    if max(img.shape[:2]) > max_img_size:
        img = sv.resize_image(img, (max_img_size,) * 2, keep_aspect_ratio=True)
    return f"data:image/jpg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")


def make_content(data: list[str | tuple[str, Any]],
                 max_img_size: int = 640):
    ret = []
    for item in data:
        if isinstance(item, str):
            item = "text", item
        if isinstance(item[1], np.ndarray):
            item = item[0], process_image(item[1], max_img_size)
        ret.append({"type": item[0], item[0]: item[1]})
    return ret
