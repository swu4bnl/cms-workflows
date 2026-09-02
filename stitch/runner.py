"""Fetch tiled runs for one stitched sample, stitch them, and save the result.

Workflow:
    1. Find the Tiled runs that belong together: either a fixed scan-id range,
       or (in "anchor mode") every run that shares a stitch_group_id with one
       anchor run.
    2. Read each run's detector image plus its stitching metadata (tile label,
       tiling mode, stage position, etc).
    3. Group tiles by detector and stitch each group into one image.
    4. Save the stitched image/json under ``out_dir`` and write an index file.
"""

import json
import os
import re
from typing import Any, Dict, List, Mapping

import numpy as np
from tiled.client import from_uri
from tiled.queries import Key

from .core.adapters.tiled_adapter import build_groups_from_tiled_runs, extract_start_doc
from .core.config import load_config
from .core.core import stitch_tiles
from .core.grouping import disambiguate_repeated_tile_groups
from .core.modes import get_required_labels
from .core.offsets import OffsetConfig
from .core.serialize import result_to_serializable, save_result_image, save_result_json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "stitching_defaults.json")
READBACK_X_KEY = "detector_x_mm"
READBACK_Y_KEY = "detector_y_mm"
KNOWN_FILE_EXTENSIONS = {".tif", ".tiff", ".npz", ".json", ".npy", ".edf"}


def safe_name(text: Any) -> str:
    """Replace characters that are unsafe in file/folder names with "_"."""
    unsafe_chars = '<>:"/\\|?* '
    result = str(text)
    for ch in unsafe_chars:
        result = result.replace(ch, "_")
    return result


def _log(logger: Any, message: str) -> None:
    """Send a status message to the Prefect logger if one was given, else print it."""
    if logger is not None:
        logger.info(message)
    else:
        print(message)


# ---------------------------------------------------------------------------
# Read one Tiled run's image + stitching metadata.
# ---------------------------------------------------------------------------

def _read_primary_data(run: Any):
    """Return a run's primary data stream (works for both live and cached Tiled runs)."""
    if hasattr(run, "primary"):
        return run.primary.read()
    return run["primary"]["data"].read()


def _detector_image_keys(run: Any, detector_configs: List[Mapping[str, Any]]) -> List[str]:
    """Return which configured detector image keys are present on this run."""
    primary = _read_primary_data(run)
    available_keys = {str(key) for key in primary.data_vars if str(key).endswith("_image")}
    if not available_keys:
        raise RuntimeError("No image-like fields found in run.primary.read().")
    return [
        str(detector["image_key"])
        for detector in detector_configs
        if str(detector.get("image_key")) in available_keys
    ]


def _read_image(run: Any, image_key: str) -> np.ndarray:
    """Read one detector's 2D image out of a run, dropping any extra singleton dimensions."""
    primary = _read_primary_data(run)
    image = np.squeeze(np.asarray(primary[image_key]))
    if image.ndim == 3:
        image = image[0]
    if image.ndim != 2:
        raise RuntimeError(f"Unsupported image array shape for key {image_key}: {image.shape}")
    return np.asarray(image, dtype=np.float64)


def _detector_config_for_image_key(image_key: str, detector_configs: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    return next((d for d in detector_configs if str(d.get("image_key")) == str(image_key)), {})


def _detector_config_for_group_id(group_id: str, detector_configs: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    for detector in detector_configs:
        name = str(detector.get("name", ""))
        if name and f"::{name}::" in str(group_id):
            return detector
    raise RuntimeError(f"No detector configuration found for stitched group {group_id!r}")


def _tile_readback_position(
    config: Mapping[str, Any], mode: str, label: str, position_prefix: str
) -> Dict[str, float]:
    """Look up one tile's stage position from the ``tiling_configs`` table in the stitch config."""
    entries = config.get("tiling_configs", {}).get(mode, [])
    entry = next((e for e in entries if str(e.get("label")) == str(label)), None)
    if entry is None:
        raise RuntimeError(f"No tiling_configs entry found for mode={mode!r}, label={label!r}")
    return {
        READBACK_X_KEY: float(entry[f"{position_prefix}x"]),
        READBACK_Y_KEY: float(entry[f"{position_prefix}y"]),
    }


def _build_tile_entry(
    run: Any,
    image_key: str,
    config: Mapping[str, Any],
    detector_configs: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Turn one (run, detector image) pair into the tile entry the stitcher expects."""
    start = extract_start_doc(run)
    detector_config = _detector_config_for_image_key(image_key, detector_configs)
    detector_name = detector_config.get("name") or image_key.replace("_image", "")
    mode = str(start.get("stitch_tiling_mode"))
    label = str(start.get("stitch_tile_label"))
    position_prefix = str(detector_config.get("position_prefix", "SAXS"))

    metadata = {
        "stitch_group_id": f"{start.get('stitch_group_id')}::{detector_name}",
        "sample_name": start.get("sample_name"),
        "stitch_tiling_mode": start.get("stitch_tiling_mode"),
        "stitch_tile_label": start.get("stitch_tile_label"),
        "stitch_tile_index": start.get("stitch_tile_index"),
        "stitch_tile_total": start.get("stitch_tile_total"),
        "detector_readback": _tile_readback_position(config, mode, label, position_prefix),
        "mask_path": detector_config.get("mask_path") or None,
        "detector_name": detector_name,
        "image_key": image_key,
        "source_uid": start.get("uid"),
        "source_scan_id": start.get("scan_id"),
        "source_filename": start.get("filename"),
    }
    return {"image": _read_image(run, image_key), "metadata": metadata, "mask": None}


def _stitched_token(detector_name: str, mode: str, output_config: Mapping[str, Any]) -> str:
    """The text that replaces a tile's position label in output names, e.g. "stitched_ygaps"."""
    template = output_config.get("filename_rule", {}).get("stitched_token_template", "stitched_{mode}")
    return template.format(detector=detector_name, mode=mode)


def _output_base_name(group_id: str, tiles: List[Any], output_config: Mapping[str, Any]) -> str:
    """Build the output filename (without extension) for one stitched group.

    Starts from the last tile's source filename, e.g. "sample_0001_pos2_saxs.tiff",
    strips the extension, then swaps the tile's position label ("pos2") for the
    stitched token ("stitched_ygaps"): "sample_0001_stitched_ygaps_saxs". If the
    label isn't found in the filename, the token is appended instead (unless
    disabled via ``append_if_tile_label_missing``). Falls back to a sanitized
    ``group_id`` if no tile has a usable source filename.
    """
    ordered_tiles = sorted(tiles, key=lambda t: (t.metadata.stitch_tile_index, t.metadata.stitch_tile_label))
    if not ordered_tiles or not ordered_tiles[-1].metadata.source_filename:
        return safe_name(group_id)

    tile = ordered_tiles[-1]
    filename = str(tile.metadata.source_filename)
    label = str(tile.metadata.stitch_tile_label)
    mode = str(tile.metadata.stitch_tiling_mode)

    stem, extension = os.path.splitext(os.path.basename(filename))
    if extension.lower() not in KNOWN_FILE_EXTENSIONS:
        stem = os.path.basename(filename)

    filename_rule = output_config.get("filename_rule", {})
    stitched_token = _stitched_token(detector_name="", mode=mode, output_config=output_config)
    replacement = filename_rule.get("replace_tile_label_with", "{stitched_token}").format(
        detector="", mode=mode, stitched_token=stitched_token
    )

    # Only replace the label where it stands alone (bounded by "_" or the string
    # edges), so e.g. label "pos1" doesn't match inside "sample_pos10".
    renamed, replaced_count = re.subn(rf"(^|_){re.escape(label)}(?=_|$)", rf"\1{replacement}", stem, count=1)
    if not replaced_count and filename_rule.get("append_if_tile_label_missing", True):
        renamed = f"{stem}_{replacement}"
    return safe_name(renamed)


def _output_paths(
    out_dir: str,
    detector_name: str,
    mode: str,
    output_base_name: str,
    output_config: Mapping[str, Any],
) -> Dict[str, str]:
    """Build the {directory, image, json} paths for one stitched group's output."""
    stitched_token = _stitched_token(detector_name, mode, output_config)
    subfolder_template = output_config.get("subfolder_template", "{detector}/{stitched_token}")
    subfolder = subfolder_template.format(detector=detector_name, mode=mode, stitched_token=stitched_token)
    subfolder_parts = [safe_name(part) for part in subfolder.replace("\\", "/").split("/") if part]
    directory = os.path.join(out_dir, *subfolder_parts)

    image_format = str(output_config.get("image_format", "tiff")).lower()
    image_extension = str(output_config.get("image_extension", ".tiff"))
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    return {
        "directory": directory,
        "image_format": image_format,
        "image": os.path.join(directory, f"{output_base_name}{image_extension}"),
        "json": os.path.join(directory, f"{output_base_name}.json"),
    }


def _open_catalog(tiled_uri: str, catalog_path: str) -> Any:
    """Connect to Tiled and step into the configured catalog path (e.g. "cms/raw")."""
    node = from_uri(tiled_uri)
    for part in catalog_path.split("/"):
        node = node[part]
    return node


def _search_by_key(node: Any, key: str, value: Any) -> List[Any]:
    """Return every run in the catalog whose metadata has ``key == value``."""
    result = node.search(Key(key) == value)
    return [result[match_key] for match_key in result]


def _fetch_runs_in_scan_range(node: Any, start_scan: int, end_scan: int) -> List[Any]:
    """Fetch every run whose scan_id falls in [start_scan, end_scan]."""
    runs: List[Any] = []
    for scan_id in range(start_scan, end_scan + 1):
        runs.extend(_search_by_key(node, "scan_id", int(scan_id)))
    return runs


def _find_anchor_run(node: Any, anchor_scan: int | None, anchor_uid: str | None) -> Any:
    """Locate the single run that anchors the stitch group, by scan_id or uid."""
    if anchor_uid is not None:
        try:
            return node[anchor_uid]
        except Exception:
            matches = _search_by_key(node, "uid", str(anchor_uid))
            if not matches:
                raise RuntimeError(f"Anchor uid={anchor_uid!r} was not found.")
            return matches[0]

    if anchor_scan is not None:
        matches = _search_by_key(node, "scan_id", int(anchor_scan))
        if not matches:
            raise RuntimeError(f"Anchor scan_id={anchor_scan} was not found.")
        if len(matches) > 1:
            raise RuntimeError(
                f"Anchor scan_id={anchor_scan} matched {len(matches)} runs; "
                "pass anchor_uid instead for a reproducible result."
            )
        return matches[0]

    raise RuntimeError("Provide either anchor_scan or anchor_uid.")


def _select_group_tiles(
    candidate_runs: List[Any],
    *,
    anchor_run: Any,
    group_id: str,
    mode: str,
    required_labels: List[str],
) -> tuple[List[Any], List[int]]:
    """Pick the complete set of required tiles from the same acquisition pass as the anchor.

    Candidate runs are bucketed by tile label and sorted oldest-to-newest. The
    anchor is the Nth run with its own label, so the matching tile for every
    other label is also taken from position N, even if earlier passes were
    incomplete.
    """
    anchor_start = extract_start_doc(anchor_run)
    anchor_scan_id = int(anchor_start["scan_id"])
    anchor_label = str(anchor_start.get("stitch_tile_label"))
    anchor_uid = anchor_start.get("uid")
    sample_name = anchor_start.get("sample_name")

    runs_by_label: Dict[str, List[Any]] = {label: [] for label in required_labels}
    for run in candidate_runs:
        start = extract_start_doc(run)
        if str(start.get("stitch_group_id")) != group_id:
            continue
        if str(start.get("stitch_tiling_mode")) != mode:
            continue
        if sample_name and start.get("sample_name") != sample_name:
            continue
        if int(start.get("scan_id", 0)) > anchor_scan_id:
            continue
        label = str(start.get("stitch_tile_label"))
        if label in required_labels:
            runs_by_label[label].append(run)

    for label in required_labels:
        runs_by_label[label].sort(key=lambda run: int(extract_start_doc(run).get("scan_id", 0)))

    anchor_position = next(
        (
            index
            for index, run in enumerate(runs_by_label.get(anchor_label, []))
            if extract_start_doc(run).get("uid") == anchor_uid
        ),
        None,
    )
    if anchor_position is None:
        raise RuntimeError(
            f"Anchor scan {anchor_scan_id} has tile label {anchor_label!r}, "
            f"which is not required for mode={mode!r}. Required labels: {required_labels}."
        )

    missing_labels = [label for label in required_labels if len(runs_by_label[label]) <= anchor_position]
    if missing_labels:
        raise RuntimeError(
            f"Could not find all required tiles for mode={mode!r} from anchor scan {anchor_scan_id}. "
            f"Missing labels: {missing_labels}."
        )

    tiles = [runs_by_label[label][anchor_position] for label in required_labels]
    scan_ids = [int(extract_start_doc(run).get("scan_id")) for run in tiles]
    return tiles, [min(scan_ids), max(scan_ids)]


def _fetch_anchor_group(
    tiled_uri: str,
    catalog_path: str,
    config_path: str,
    anchor_scan: int | None,
    anchor_uid: str | None,
    max_lookback: int,
    logger: Any = None,
) -> tuple[List[Any], List[int]]:
    """Given one anchor run, fetch the sibling runs that complete its stitch group.

    Tries an indexed search by ``stitch_group_id`` first; if that search isn't
    supported by the catalog (or finds nothing), falls back to scanning backward
    through ``scan_id`` for up to ``max_lookback`` scans.
    """
    node = _open_catalog(tiled_uri, catalog_path)
    anchor_run = _find_anchor_run(node, anchor_scan, anchor_uid)
    anchor_start = extract_start_doc(anchor_run)

    group_id = anchor_start.get("stitch_group_id")
    mode = anchor_start.get("stitch_tiling_mode")
    scan_id = anchor_start.get("scan_id")
    if group_id is None or mode is None or scan_id is None:
        raise RuntimeError(
            "Anchor run is missing required metadata. Need stitch_group_id, stitch_tiling_mode, and scan_id."
        )

    required_labels = get_required_labels(str(mode), config_path=config_path)
    anchor_label = str(anchor_start.get("stitch_tile_label"))
    if anchor_label != required_labels[-1]:
        raise RuntimeError(
            f"Could not find all required tiles for mode={mode!r} from anchor scan {scan_id}. "
            f"Anchor label {anchor_label!r} is not the final required tile {required_labels[-1]!r}."
        )

    try:
        candidate_runs = _search_by_key(node, "stitch_group_id", str(group_id))
    except Exception as exc:
        _log(logger, f"stitch_group_id search unavailable ({exc}); falling back to scan_id lookback.")
        candidate_runs = []

    if not candidate_runs:
        lookback = max(int(max_lookback), 1)
        candidate_runs = _fetch_runs_in_scan_range(node, int(scan_id) - lookback + 1, int(scan_id))

    tiles, scan_range = _select_group_tiles(
        candidate_runs,
        anchor_run=anchor_run,
        group_id=str(group_id),
        mode=str(mode),
        required_labels=required_labels,
    )
    _log(logger, f"Anchor group resolved: group_id={group_id} mode={mode} scans={scan_range}")
    return tiles, scan_range


def run_stitch_validation(
    *,
    start_scan: int | None = None,
    end_scan: int | None = None,
    anchor_scan: int | None = None,
    anchor_uid: str | None = None,
    max_lookback: int = 50,
    tiled_uri: str = "https://tiled.nsls2.bnl.gov",
    catalog_path: str = "cms/raw",
    config_path: str | None = None,
    out_dir: str | None = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Fetch tile runs from Tiled, stitch each detector group, and write outputs.

    This is the core entrypoint used by the Prefect workflow. In anchor mode,
    pass ``anchor_scan`` or ``anchor_uid`` and the function finds the complete
    tile group around that anchor. For manual range validation, callers may
    instead pass a fixed ``start_scan``/``end_scan`` range.

    Output files are written under ``out_dir`` using the configured output rule.
    With the default config, detector folders look like ``maxs/stitched_ygaps``.
    The returned dict includes the absolute output directory, validation index,
    resolved scan range, and stitched group count.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    out_dir = out_dir or os.path.join("outputs", "phase1_validation")

    config = load_config(config_path)
    detector_configs = config.get("detector", {}).get("image_streams", [])
    output_config = config.get("outputs", {})
    os.makedirs(out_dir, exist_ok=True)

    if anchor_scan is not None or anchor_uid is not None:
        runs, resolved_scan_range = _fetch_anchor_group(
            tiled_uri=tiled_uri,
            catalog_path=catalog_path,
            config_path=config_path,
            anchor_scan=anchor_scan,
            anchor_uid=anchor_uid,
            max_lookback=max_lookback,
            logger=logger,
        )
    else:
        if start_scan is None or end_scan is None:
            raise RuntimeError("Provide start_scan and end_scan, or use anchor_scan/anchor_uid.")
        node = _open_catalog(tiled_uri, catalog_path)
        runs = _fetch_runs_in_scan_range(node, start_scan, end_scan)
        resolved_scan_range = [start_scan, end_scan]

    if not runs:
        raise RuntimeError("No runs fetched from Tiled. Check catalog path, auth, and selection arguments.")

    tile_entries: List[Dict[str, Any]] = []
    for run in runs:
        start = extract_start_doc(run)
        if not start.get("stitch_group_id"):
            continue
        for image_key in _detector_image_keys(run, detector_configs):
            tile_entries.append(_build_tile_entry(run, image_key, config, detector_configs))

    if not tile_entries:
        raise RuntimeError("No runs with stitch_group_id found in selected scan range.")
    _log(logger, f"Read stitch images: runs={len(runs)} tile_entries={len(tile_entries)}")

    tile_entries = disambiguate_repeated_tile_groups(tile_entries)
    grouped_tiles = build_groups_from_tiled_runs(
        tile_entries,
        image_loader=lambda entry: np.asarray(entry["image"], dtype=np.float64),
    )
    _log(logger, f"Grouped stitch tiles: groups={len(grouped_tiles)}")

    index_payload = {
        "scan_range": resolved_scan_range,
        "tiled_uri": tiled_uri,
        "catalog_path": catalog_path,
        "settings": {
            "offsets": {"source": "tiling_configs"},
            "coordinate_system": config.get("coordinate_system", {}),
            "detector_image_streams": detector_configs,
            "outputs": output_config,
        },
        "group_count": len(grouped_tiles),
        "groups": [],
    }

    for group_id, tiles in grouped_tiles.items():
        detector_config = _detector_config_for_group_id(group_id, detector_configs)
        offset_config = OffsetConfig(
            pixel_size_um=float(detector_config["pixel_size_um"]),
            readback_x_key=READBACK_X_KEY,
            readback_y_key=READBACK_Y_KEY,
            x_sign=int(config.get("coordinate_system", {}).get("x_sign", 1)),
            y_sign=int(config.get("coordinate_system", {}).get("y_sign", 1)),
        )
        stitched = stitch_tiles(tiles, offset_config)
        serialized = result_to_serializable(stitched)

        mode = str(tiles[0].metadata.stitch_tiling_mode)
        detector_name = str(detector_config.get("name") or "unknown_detector")
        output_base_name = _output_base_name(group_id, tiles, output_config)
        output_paths = _output_paths(out_dir, detector_name, mode, output_base_name, output_config)

        os.makedirs(output_paths["directory"], exist_ok=True)
        save_result_json(stitched, output_paths["json"])
        save_result_image(stitched, output_paths["image"], output_paths["image_format"])

        index_payload["groups"].append(
            {
                "group_id": group_id,
                "detector": detector_config.get("name"),
                "pixel_size_um": detector_config.get("pixel_size_um"),
                "tile_count": len(tiles),
                "json": output_paths["json"],
                "image": output_paths["image"],
                "image_format": output_paths["image_format"],
                "output_base_name": output_base_name,
                "qa_flags": serialized["result_metadata"].get("qa_flags", []),
            }
        )
        _log(
            logger,
            f"Stitched group={group_id} detector={detector_name} tiles={len(tiles)} output={output_paths['image']}",
        )

    index_path = os.path.join(out_dir, "validation_index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2)

    _log(
        logger,
        f"Stitch validation complete: groups={len(grouped_tiles)} scan_range={resolved_scan_range} "
        f"index={os.path.abspath(index_path)}",
    )
    return {
        "output_dir": os.path.abspath(out_dir),
        "index_path": os.path.abspath(index_path),
        "scan_range": resolved_scan_range,
        "group_count": len(grouped_tiles),
    }
