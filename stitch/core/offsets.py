"""Compute pixel offsets from detector readback values."""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .errors import MetadataMismatchError
from .models import TileEntry


@dataclass(frozen=True)
class OffsetConfig:
    pixel_size_um: float
    readback_x_key: str
    readback_y_key: str
    x_bias_px: float = 0.0
    y_bias_px: float = 0.0
    x_sign: int = 1
    y_sign: int = 1


def _mm_per_pixel(pixel_size_um: float) -> float:
    return float(pixel_size_um) / 1000.0


def compute_pixel_offsets(
    tiles: Iterable[TileEntry],
    cfg: OffsetConfig,
) -> Dict[str, Tuple[int, int]]:
    ordered = sorted(
        list(tiles), key=lambda t: (t.metadata.stitch_tile_index, t.metadata.stitch_tile_label)
    )
    if not ordered:
        return {}

    mm_per_pixel = _mm_per_pixel(cfg.pixel_size_um)
    if mm_per_pixel <= 0:
        raise MetadataMismatchError("pixel_size_um must be > 0")

    ref_md = ordered[0].metadata.detector_readback
    if cfg.readback_x_key not in ref_md or cfg.readback_y_key not in ref_md:
        raise MetadataMismatchError(
            f"Reference tile missing readback keys {cfg.readback_x_key}/{cfg.readback_y_key}."
        )

    x0 = float(ref_md[cfg.readback_x_key])
    y0 = float(ref_md[cfg.readback_y_key])

    offsets: Dict[str, Tuple[int, int]] = {}
    for tile in ordered:
        md = tile.metadata.detector_readback
        if cfg.readback_x_key not in md or cfg.readback_y_key not in md:
            raise MetadataMismatchError(
                f"Tile {tile.metadata.stitch_tile_label} missing readback keys "
                f"{cfg.readback_x_key}/{cfg.readback_y_key}."
            )

        dx_mm = float(md[cfg.readback_x_key]) - x0
        dy_mm = float(md[cfg.readback_y_key]) - y0
        dx_px = int(round(cfg.x_sign * dx_mm / mm_per_pixel + cfg.x_bias_px))
        dy_px = int(round(cfg.y_sign * dy_mm / mm_per_pixel + cfg.y_bias_px))

        offsets[tile.metadata.stitch_tile_label] = (dx_px, dy_px)

    return offsets
