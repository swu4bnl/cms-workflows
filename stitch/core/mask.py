import os
from typing import Optional
import numpy as np
from .errors import MissingMaskError

def _load_mask_from_path(mask_path: str) -> np.ndarray:
    if not os.path.isfile(mask_path):
        raise MissingMaskError(f"Mask file not found: {mask_path}")
    from PIL import Image
    data = np.asarray(Image.open(mask_path).convert("I"), dtype=np.float64)
    if data.size == 0:
        raise MissingMaskError(f"Mask file is empty: {mask_path}")
    if data.max() > 0:
        data = data / data.max()
    return (data > 0.5).astype(np.float64)


def resolve_mask(mask: Optional[np.ndarray], mask_path: Optional[str]) -> np.ndarray:
    if mask is not None:
        return (np.asarray(mask, dtype=np.float64) > 0.5).astype(np.float64)
    if mask_path:
        return _load_mask_from_path(mask_path)
    raise MissingMaskError("Mask is required for every tile (array or mask_path).")


def validate_mask_shape(mask: np.ndarray, image: np.ndarray) -> None:
    if mask.shape != image.shape:
        raise MissingMaskError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    validate_mask_shape(mask, image)
    return np.asarray(image, dtype=np.float64) * np.asarray(mask, dtype=np.float64)
