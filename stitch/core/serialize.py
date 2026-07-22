"""Serialization helpers for stitch results."""

import json
from typing import Any, Dict

import numpy as np

from .models import StitchResult


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def result_to_serializable(result: StitchResult) -> Dict[str, Any]:
    image = result.stitched_image
    md = _to_native(result.result_metadata)

    return {
        "result_metadata": md,
        "image_summary": {
            "shape": [int(image.shape[0]), int(image.shape[1])],
            "dtype": str(image.dtype),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
            "mean": float(np.mean(image)),
        },
    }


def save_result_json(result: StitchResult, out_path: str) -> None:
    payload = result_to_serializable(result)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_result_tiff(result: StitchResult, out_path: str) -> None:
    from PIL import Image

    image = np.asarray(result.stitched_image, dtype=np.float32)
    Image.fromarray(image).save(out_path, format="TIFF")


def save_result_image(result: StitchResult, out_path: str, image_format: str) -> None:
    fmt = str(image_format).lower()
    if fmt in {"tif", "tiff"}:
        save_result_tiff(result, out_path)
        return
    if fmt == "npz":
        save_result_npz(result, out_path)
        return
    raise ValueError(f"Unsupported output image_format: {image_format!r}")


def save_result_npz(result: StitchResult, out_path: str) -> None:
    payload = json.dumps(result_to_serializable(result))
    np.savez_compressed(out_path, stitched_image=result.stitched_image, summary_json=payload)
