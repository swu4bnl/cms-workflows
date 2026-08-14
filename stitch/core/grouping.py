"""Grouping and normalization helpers for tile entries."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .errors import MetadataMismatchError, MissingTileError
from .models import TileEntry, TileMetadata
from .modes import validate_mode_labels


def normalize_tile_metadata(raw: Mapping[str, Any]) -> TileMetadata:
    group_id = raw.get("stitch_group_id") or raw.get("group_id")
    mode = raw.get("stitch_tiling_mode") or raw.get("tiling_mode")
    tile_label = raw.get("stitch_tile_label") or raw.get("tile_label") or raw.get("detector_position")
    tile_index = raw.get("stitch_tile_index") or raw.get("tile_index")

    if group_id is None or mode is None or tile_label is None or tile_index is None:
        raise MetadataMismatchError(
            "Required metadata missing. Need stitch_group_id, stitch_tiling_mode, "
            "stitch_tile_label, stitch_tile_index."
        )

    detector_readback = raw.get("detector_readback") or raw.get("motor_positions") or {}
    if not isinstance(detector_readback, dict):
        raise MetadataMismatchError("detector_readback must be a dictionary")

    return TileMetadata(
        stitch_group_id=str(group_id),
        stitch_tiling_mode=str(mode),
        stitch_tile_label=str(tile_label),
        stitch_tile_index=int(tile_index),
        stitch_tile_total=(int(raw["stitch_tile_total"]) if raw.get("stitch_tile_total") is not None else None),
        detector_readback={str(k): float(v) for k, v in detector_readback.items()},
        mask_path=(str(raw["mask_path"]) if raw.get("mask_path") else None),
        source_uid=(str(raw["source_uid"]) if raw.get("source_uid") else None),
        source_scan_id=(int(raw["source_scan_id"]) if raw.get("source_scan_id") is not None else None),
        source_filename=(str(raw["source_filename"]) if raw.get("source_filename") else None),
    )


def normalize_tile_entry(raw: Mapping[str, Any]) -> TileEntry:
    if "image" not in raw:
        raise MetadataMismatchError("Tile entry is missing image array")
    if "metadata" not in raw:
        raise MetadataMismatchError("Tile entry is missing metadata")

    image = np.asarray(raw["image"], dtype=np.float64)
    md = normalize_tile_metadata(raw["metadata"])
    mask = raw.get("mask")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)

    return TileEntry(image=image, metadata=md, mask=mask)


def group_tile_entries(entries: Iterable[Mapping[str, Any]]) -> Dict[str, List[TileEntry]]:
    grouped: Dict[str, List[TileEntry]] = defaultdict(list)

    for raw in entries:
        tile = normalize_tile_entry(raw)
        grouped[tile.metadata.stitch_group_id].append(tile)

    for group_id, tiles in grouped.items():
        _validate_group(group_id, tiles)

    return dict(grouped)


def disambiguate_repeated_tile_groups(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Split repeated acquisitions into deterministic virtual groups.

    Entries are grouped by stitch_group_id plus sample name when available.
    If repeated acquisitions are present, every tile index must appear the same
    number of times; otherwise a missing-tile error is raised instead of
    reusing another tile implicitly.
    """
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        metadata = dict(entry.get("metadata", {}))
        group_id = metadata.get("stitch_group_id")
        sample_name = metadata.get("sample_name") or "unknown_sample"
        grouped[f"{group_id}::{sample_name}"].append(entry)

    output: List[Dict[str, Any]] = []
    for key, group_entries in grouped.items():
        by_tile_index: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for entry in group_entries:
            metadata = dict(entry.get("metadata", {}))
            tile_index = int(metadata.get("stitch_tile_index", 0))
            by_tile_index[tile_index].append(entry)

        counts = {tile_index: len(items) for tile_index, items in by_tile_index.items()}
        if not counts:
            continue

        repetition_count = max(counts.values())
        if any(count != repetition_count for count in counts.values()):
            raise MissingTileError(
                f"Repeated group {key} has inconsistent tile counts by index: {counts}"
            )

        for items in by_tile_index.values():
            items.sort(key=lambda entry: int(entry.get("metadata", {}).get("source_scan_id", 0)))

        has_repetitions = repetition_count > 1
        for rep_index in range(repetition_count):
            group_name = f"{key}__rep{rep_index + 1}" if has_repetitions else key
            for tile_index in sorted(by_tile_index):
                chosen = by_tile_index[tile_index][rep_index]
                metadata = dict(chosen.get("metadata", {}))
                metadata["stitch_group_id"] = group_name
                output.append(
                    {
                        "image": chosen["image"],
                        "metadata": metadata,
                        "mask": chosen.get("mask"),
                    }
                )

    return output


def _validate_group(group_id: str, tiles: List[TileEntry]) -> None:
    if not tiles:
        raise MetadataMismatchError(f"Group {group_id} is empty")

    mode_set = {t.metadata.stitch_tiling_mode for t in tiles}
    if len(mode_set) != 1:
        raise MetadataMismatchError(
            f"Group {group_id} has mixed tiling modes: {sorted(mode_set)}"
        )

    mode = next(iter(mode_set))
    labels = [t.metadata.stitch_tile_label for t in tiles]
    validate_mode_labels(mode, labels)

    # Ensure tile indices are unique and deterministic.
    idx_set = {t.metadata.stitch_tile_index for t in tiles}
    if len(idx_set) != len(tiles):
        raise MetadataMismatchError(
            f"Group {group_id} has duplicate stitch_tile_index values"
        )
