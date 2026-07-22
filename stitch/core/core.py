"""Stitcher core: array + normalized metadata only."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .errors import MetadataMismatchError
from .mask import apply_mask, resolve_mask
from .models import StitchResult, TileEntry
from .offsets import OffsetConfig, compute_pixel_offsets


@dataclass(frozen=True)
class StitchProcessingConfig:
    refine_offset_search_px: int = 0
    feather_edge_px: int = 0
    match_gain: bool = False
    destripe_rows: bool = False


def stitch_tiles(
    tiles: Iterable[TileEntry],
    offset_config: OffsetConfig,
    default_mask_path: Optional[str] = None,
    processing: Optional[StitchProcessingConfig] = None,
) -> StitchResult:
    """Stitch one tile group into a single image deterministically."""
    ordered = sorted(
        list(tiles), key=lambda t: (t.metadata.stitch_tile_index, t.metadata.stitch_tile_label)
    )
    if not ordered:
        raise MetadataMismatchError("Cannot stitch an empty tile list")

    _validate_shared_group_metadata(ordered)

    offsets = compute_pixel_offsets(ordered, offset_config)
    processing_cfg = processing or StitchProcessingConfig()

    if processing_cfg.refine_offset_search_px > 0:
        offsets = _refine_offsets_local(
            ordered,
            offsets,
            default_mask_path,
            search_radius=int(processing_cfg.refine_offset_search_px),
        )

    stitched, coverage = _accumulate_tiles(
        ordered,
        offsets,
        default_mask_path,
        feather_edge_px=int(processing_cfg.feather_edge_px),
        match_gain=bool(processing_cfg.match_gain),
        destripe_rows=bool(processing_cfg.destripe_rows),
    )

    first_md = ordered[0].metadata
    qa_flags = _build_qa_flags(coverage)
    result_metadata: Dict[str, Any] = {
        "stitch_group_id": first_md.stitch_group_id,
        "stitch_tiling_mode": first_md.stitch_tiling_mode,
        "tile_labels": [t.metadata.stitch_tile_label for t in ordered],
        "tile_indices": [t.metadata.stitch_tile_index for t in ordered],
        "offsets_used_pixels": {
            label: {"dx": int(dx), "dy": int(dy)}
            for label, (dx, dy) in offsets.items()
        },
        "processing": {
            "refine_offset_search_px": int(processing_cfg.refine_offset_search_px),
            "feather_edge_px": int(processing_cfg.feather_edge_px),
            "match_gain": bool(processing_cfg.match_gain),
            "destripe_rows": bool(processing_cfg.destripe_rows),
        },
        "qa_flags": qa_flags,
    }

    return StitchResult(stitched_image=stitched, result_metadata=result_metadata)


def _validate_shared_group_metadata(tiles: List[TileEntry]) -> None:
    group_ids = {t.metadata.stitch_group_id for t in tiles}
    if len(group_ids) != 1:
        raise MetadataMismatchError(f"Mixed stitch_group_id values: {sorted(group_ids)}")

    modes = {t.metadata.stitch_tiling_mode for t in tiles}
    if len(modes) != 1:
        raise MetadataMismatchError(f"Mixed stitch_tiling_mode values: {sorted(modes)}")


def _accumulate_tiles(
    tiles: List[TileEntry],
    offsets: Mapping[str, Tuple[int, int]],
    default_mask_path: Optional[str],
    feather_edge_px: int = 0,
    match_gain: bool = False,
    destripe_rows: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    extents = _canvas_extents(tiles, offsets)
    min_x, min_y, width, height = extents

    signal = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)

    resolved_masks: Dict[str, np.ndarray] = {}
    for tile in tiles:
        resolved_masks[tile.metadata.stitch_tile_label] = resolve_mask(
            tile.mask,
            tile.metadata.mask_path or default_mask_path,
        )

    gain_scales = _estimate_gain_scales(tiles, offsets, resolved_masks) if match_gain else {}

    for tile in tiles:
        label = tile.metadata.stitch_tile_label
        dx, dy = offsets[label]
        x0 = dx - min_x
        y0 = dy - min_y

        mask = resolved_masks[label]
        image = np.asarray(tile.image, dtype=np.float64)
        if destripe_rows:
            image = _destripe_rows(image)
        if label in gain_scales:
            image = image * gain_scales[label]

        masked = apply_mask(image, mask)
        weight_map = np.asarray(mask, dtype=np.float64)
        if feather_edge_px > 0:
            weight_map = weight_map * _feather_window(mask.shape, feather_edge_px)

        h, w = masked.shape
        signal[y0 : y0 + h, x0 : x0 + w] += masked
        weight[y0 : y0 + h, x0 : x0 + w] += weight_map

    stitched = np.zeros_like(signal)
    valid = weight > 0
    stitched[valid] = signal[valid] / weight[valid]

    return stitched, weight


def _canvas_extents(
    tiles: List[TileEntry],
    offsets: Mapping[str, Tuple[int, int]],
) -> Tuple[int, int, int, int]:
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0

    for tile in tiles:
        label = tile.metadata.stitch_tile_label
        dx, dy = offsets[label]
        h, w = tile.image.shape

        min_x = min(min_x, dx)
        min_y = min(min_y, dy)
        max_x = max(max_x, dx + w)
        max_y = max(max_y, dy + h)

    return min_x, min_y, max_x - min_x, max_y - min_y


def _build_qa_flags(weight: np.ndarray) -> List[str]:
    total_pixels = int(weight.size)
    empty_pixels = int(np.sum(weight <= 0))
    overlap_pixels = int(np.sum(weight > 1))

    flags: List[str] = []
    if empty_pixels > 0:
        frac = empty_pixels / max(total_pixels, 1)
        flags.append(f"empty_pixels_fraction={frac:.6f}")
    flags.append(f"overlap_pixels={overlap_pixels}")

    return flags


def _destripe_rows(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return image

    # Row-wise low percentile tracks additive horizontal banding robustly.
    row_bg = np.nanpercentile(np.where(finite, image, np.nan), 20.0, axis=1)
    global_bg = float(np.nanmedian(row_bg))
    correction = row_bg - global_bg

    out = np.array(image, dtype=np.float64, copy=True)
    out -= correction[:, None]
    return out


def _feather_window(shape: Tuple[int, int], edge_px: int) -> np.ndarray:
    h, w = shape
    if edge_px <= 0:
        return np.ones((h, w), dtype=np.float64)

    edge_y = min(edge_px, max(h // 2, 1))
    edge_x = min(edge_px, max(w // 2, 1))

    wy = np.ones(h, dtype=np.float64)
    wx = np.ones(w, dtype=np.float64)

    ramp_y = np.linspace(0.1, 1.0, edge_y, endpoint=True)
    ramp_x = np.linspace(0.1, 1.0, edge_x, endpoint=True)

    wy[:edge_y] = np.minimum(wy[:edge_y], ramp_y)
    wy[-edge_y:] = np.minimum(wy[-edge_y:], ramp_y[::-1])
    wx[:edge_x] = np.minimum(wx[:edge_x], ramp_x)
    wx[-edge_x:] = np.minimum(wx[-edge_x:], ramp_x[::-1])

    return np.outer(wy, wx)


def _refine_offsets_local(
    tiles: List[TileEntry],
    offsets: Mapping[str, Tuple[int, int]],
    default_mask_path: Optional[str],
    search_radius: int,
) -> Dict[str, Tuple[int, int]]:
    if search_radius <= 0 or len(tiles) < 2:
        return {k: (int(v[0]), int(v[1])) for k, v in offsets.items()}

    refined = {k: (int(v[0]), int(v[1])) for k, v in offsets.items()}

    anchor = tiles[0]
    anchor_label = anchor.metadata.stitch_tile_label
    anchor_img = np.asarray(anchor.image, dtype=np.float64)
    anchor_mask = resolve_mask(anchor.mask, anchor.metadata.mask_path or default_mask_path)
    anchor_pos = refined[anchor_label]

    for tile in tiles[1:]:
        label = tile.metadata.stitch_tile_label
        base_dx, base_dy = refined[label]
        tile_img = np.asarray(tile.image, dtype=np.float64)
        tile_mask = resolve_mask(tile.mask, tile.metadata.mask_path or default_mask_path)

        best = (base_dx, base_dy)
        best_score = np.inf

        for sx in range(-search_radius, search_radius + 1):
            for sy in range(-search_radius, search_radius + 1):
                cand_dx = base_dx + sx
                cand_dy = base_dy + sy
                score = _overlap_difference_score(
                    anchor_img,
                    anchor_mask,
                    anchor_pos,
                    tile_img,
                    tile_mask,
                    (cand_dx, cand_dy),
                )
                if score < best_score:
                    best_score = score
                    best = (cand_dx, cand_dy)

        refined[label] = best

    return refined


def _overlap_difference_score(
    a_img: np.ndarray,
    a_mask: np.ndarray,
    a_pos: Tuple[int, int],
    b_img: np.ndarray,
    b_mask: np.ndarray,
    b_pos: Tuple[int, int],
) -> float:
    ax, ay = a_pos
    bx, by = b_pos
    ah, aw = a_img.shape
    bh, bw = b_img.shape

    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)

    if x1 <= x0 or y1 <= y0:
        return np.inf

    a_x0 = x0 - ax
    a_y0 = y0 - ay
    b_x0 = x0 - bx
    b_y0 = y0 - by
    h = y1 - y0
    w = x1 - x0

    a_patch = a_img[a_y0 : a_y0 + h, a_x0 : a_x0 + w]
    b_patch = b_img[b_y0 : b_y0 + h, b_x0 : b_x0 + w]
    a_valid = a_mask[a_y0 : a_y0 + h, a_x0 : a_x0 + w] > 0
    b_valid = b_mask[b_y0 : b_y0 + h, b_x0 : b_x0 + w] > 0
    valid = a_valid & b_valid & np.isfinite(a_patch) & np.isfinite(b_patch)

    if int(np.count_nonzero(valid)) < 256:
        return np.inf

    a_vals = np.log1p(np.abs(a_patch[valid]))
    b_vals = np.log1p(np.abs(b_patch[valid]))

    # Remove local level bias before scoring alignment quality.
    a_vals = a_vals - float(np.median(a_vals))
    b_vals = b_vals - float(np.median(b_vals))
    return float(np.mean(np.abs(a_vals - b_vals)))


def _estimate_gain_scales(
    tiles: List[TileEntry],
    offsets: Mapping[str, Tuple[int, int]],
    masks: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    if len(tiles) < 2:
        return {}

    anchor = tiles[0]
    anchor_label = anchor.metadata.stitch_tile_label
    anchor_img = np.asarray(anchor.image, dtype=np.float64)
    anchor_mask = masks[anchor_label]
    anchor_pos = offsets[anchor_label]

    scales: Dict[str, float] = {}
    for tile in tiles[1:]:
        label = tile.metadata.stitch_tile_label
        ratio = _overlap_gain_ratio(
            anchor_img,
            anchor_mask,
            anchor_pos,
            np.asarray(tile.image, dtype=np.float64),
            masks[label],
            offsets[label],
        )
        if ratio is None:
            continue
        scales[label] = float(np.clip(ratio, 0.5, 2.0))

    return scales


def _overlap_gain_ratio(
    a_img: np.ndarray,
    a_mask: np.ndarray,
    a_pos: Tuple[int, int],
    b_img: np.ndarray,
    b_mask: np.ndarray,
    b_pos: Tuple[int, int],
) -> Optional[float]:
    ax, ay = a_pos
    bx, by = b_pos
    ah, aw = a_img.shape
    bh, bw = b_img.shape

    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None

    a_x0 = x0 - ax
    a_y0 = y0 - ay
    b_x0 = x0 - bx
    b_y0 = y0 - by
    h = y1 - y0
    w = x1 - x0

    a_patch = a_img[a_y0 : a_y0 + h, a_x0 : a_x0 + w]
    b_patch = b_img[b_y0 : b_y0 + h, b_x0 : b_x0 + w]
    a_valid = a_mask[a_y0 : a_y0 + h, a_x0 : a_x0 + w] > 0
    b_valid = b_mask[b_y0 : b_y0 + h, b_x0 : b_x0 + w] > 0
    valid = a_valid & b_valid & np.isfinite(a_patch) & np.isfinite(b_patch)

    if int(np.count_nonzero(valid)) < 256:
        return None

    eps = 1e-6
    a_vals = np.abs(a_patch[valid]) + eps
    b_vals = np.abs(b_patch[valid]) + eps
    denom = float(np.median(b_vals))
    if denom <= 0:
        return None
    return float(np.median(a_vals) / denom)
