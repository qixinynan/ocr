from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import cv2
import easyocr
import numpy as np
import torch


DEFAULT_LANGS = ["ch_sim", "en"]

# The NVIDIA vLLM 26.05 base image can produce incorrect EasyOCR CRAFT boxes
# through cuDNN on GB10. CUDA without cuDNN gives correct results.
torch.backends.cudnn.enabled = False


@lru_cache(maxsize=4)
def get_reader(lang_key: str) -> easyocr.Reader:
    langs = [item for item in lang_key.split(",") if item]
    if not langs:
        langs = DEFAULT_LANGS

    gpu = torch.cuda.is_available()
    return easyocr.Reader(
        langs,
        gpu=gpu,
        model_storage_directory=os.environ.get("EASYOCR_MODULE_PATH", "/models/easyocr"),
        download_enabled=True,
    )


def recognize(image_bytes: bytes, langs: list[str] | None = None) -> dict[str, Any]:
    lang_key = ",".join(langs or DEFAULT_LANGS)
    reader = get_reader(lang_key)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")

    rows = reader.readtext(image)

    results = []
    for box, text, confidence in rows:
        if not text:
            continue
        results.append(
            {
                "text": text,
                "confidence": float(confidence),
                "box": [[float(x), float(y)] for x, y in box],
            }
        )

    return {
        "engine": "easyocr",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "languages": lang_key.split(","),
        "results": results,
    }
