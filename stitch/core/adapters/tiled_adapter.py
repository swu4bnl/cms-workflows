"""Tiled adapter utilities.

This module keeps Tiled retrieval outside the core stitcher. The stitcher only
receives image arrays and normalized metadata dictionaries.
"""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..grouping import group_tile_entries


def _extract_start_doc(run: Any) -> Dict[str, Any]:
    if hasattr(run, "start"):
        try:
            return dict(run.start)
        except Exception:
            pass

    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, Mapping):
        start = metadata.get("start")
        if isinstance(start, Mapping):
            return dict(start)

    if isinstance(run, Mapping):
        metadata = run.get("metadata", {})
        if isinstance(metadata, Mapping):
            start = metadata.get("start")
            if isinstance(start, Mapping):
                return dict(start)
            return dict(metadata)

    return {}


def _pick_run_value(run: Any, name: str, default: Any = None) -> Any:
    if isinstance(run, Mapping):
        return run.get(name, default)
    return getattr(run, name, default)


def normalize_tiled_run(
    run: Mapping[str, Any],
    image_loader: Callable[[Mapping[str, Any]], np.ndarray],
    metadata_key_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert one Tiled run to a normalized tile entry."""
    key_map = metadata_key_map or {}
    raw_md = _extract_start_doc(run)

    def pick(name: str, default: Any = None) -> Any:
        source_key = key_map.get(name, name)
        return _pick_run_value(run, source_key, default)

    md = {
        "stitch_group_id": raw_md.get(key_map.get("stitch_group_id", "stitch_group_id")),
        "stitch_tiling_mode": raw_md.get(key_map.get("stitch_tiling_mode", "stitch_tiling_mode")),
        "stitch_tile_label": raw_md.get(key_map.get("stitch_tile_label", "stitch_tile_label")),
        "stitch_tile_index": raw_md.get(key_map.get("stitch_tile_index", "stitch_tile_index")),
        "stitch_tile_total": raw_md.get(key_map.get("stitch_tile_total", "stitch_tile_total")),
        "detector_readback": raw_md.get(key_map.get("detector_readback", "detector_readback"), {}),
        "mask_path": raw_md.get(key_map.get("mask_path", "mask_path")),
        "source_uid": pick("uid"),
        "source_scan_id": raw_md.get(key_map.get("scan_id", "scan_id")),
        "source_filename": raw_md.get(key_map.get("filename", "filename")) or raw_md.get("source_filename"),
    }

    return {
        "image": image_loader(run),
        "metadata": md,
        "mask": None,
    }


def build_groups_from_tiled_runs(
    runs: Iterable[Mapping[str, Any]],
    image_loader: Callable[[Mapping[str, Any]], np.ndarray],
    metadata_key_map: Optional[Dict[str, str]] = None,
):
    """Normalize Tiled runs and group them by stitch_group_id."""
    entries: List[Dict[str, Any]] = []
    for run in runs:
        entries.append(normalize_tiled_run(run, image_loader, metadata_key_map))
    return group_tile_entries(entries)
