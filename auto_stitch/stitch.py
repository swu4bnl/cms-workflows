import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tiled.client import from_uri
from tiled.queries import Key

from stitching.adapters.tiled_adapter import build_groups_from_tiled_runs
from stitching.config import load_config
from stitching.core import stitch_tiles
from stitching.grouping import disambiguate_repeated_tile_groups
from stitching.modes import get_required_labels
from stitching.offsets import OffsetConfig
from stitching.serialize import result_to_serializable, save_result_image, save_result_json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "stitching_defaults.json")
FALLBACK_OUTPUT_DIR = "StitchingOutputs"
FALLBACK_INDEX_PATH = os.path.join(FALLBACK_OUTPUT_DIR, "validation_index.json")
READBACK_X_KEY = "detector_x_mm"
READBACK_Y_KEY = "detector_y_mm"


def load_settings(config_path=None):
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def config_value(settings, section, key, default=None):
    value = settings.get(section, {}).get(key, default)
    return default if value is None else value


def output_dir_from_config(settings):
    return config_value(settings, "outputs", "directory", FALLBACK_OUTPUT_DIR)


def index_path_from_config(settings):
    output_dir = output_dir_from_config(settings)
    index_filename = config_value(settings, "outputs", "index_filename", "validation_index.json")
    return os.path.join(output_dir, index_filename)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def safe_name(text):
    bad = '<>:"/\\|?* '
    out = str(text)
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def _read_primary(run: Any):
    if hasattr(run, "primary"):
        return run.primary.read()
    return run["primary"]["data"].read()


def _image_keys(run: Any, detector_configs: List[Mapping[str, Any]]) -> List[str]:
    primary = _read_primary(run)
    candidates = [str(key) for key in primary.data_vars if str(key).endswith("_image")]
    if not candidates:
        raise RuntimeError("No image-like fields found in run.primary.read().")

    configured = [str(detector_config.get("image_key")) for detector_config in detector_configs]
    return [image_key for image_key in configured if image_key in candidates]


def _extract_image_array(run: Any, image_key: str) -> np.ndarray:
    primary = _read_primary(run)

    arr = np.asarray(primary[image_key])
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return np.asarray(arr[0], dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64)

    raise RuntimeError(f"Unsupported image array shape for key {image_key}: {arr.shape}")


def _detector_config_for_image_key(image_key: str, detector_configs: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    for detector_config in detector_configs:
        if str(detector_config.get("image_key")) == str(image_key):
            return detector_config
    return {}


def _detector_config_for_group_id(group_id: str, detector_configs: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    for detector_config in detector_configs:
        name = str(detector_config.get("name", ""))
        if name and f"::{name}::" in str(group_id):
            return detector_config
    raise RuntimeError(f"No detector configuration found for stitched group {group_id!r}")


def _tile_config_by_label(config: Mapping[str, Any], mode: str, label: str) -> Mapping[str, Any]:
    entries = config.get("tiling_configs", {}).get(mode, [])
    for entry in entries:
        if str(entry.get("label")) == str(label):
            return entry
    raise RuntimeError(f"No tiling_configs entry found for mode={mode!r}, label={label!r}")


def _readback_from_tiling_config(
    start: Mapping[str, Any],
    config: Mapping[str, Any],
    detector_config: Mapping[str, Any],
) -> Dict[str, float]:
    mode = str(start.get("stitch_tiling_mode"))
    label = str(start.get("stitch_tile_label"))
    prefix = str(detector_config.get("position_prefix", "SAXS"))
    tile_config = _tile_config_by_label(config, mode, label)
    return {
        READBACK_X_KEY: float(tile_config[f"{prefix}x"]),
        READBACK_Y_KEY: float(tile_config[f"{prefix}y"]),
    }


def _normalize_run(
    run: Any,
    image_key: str,
    config: Mapping[str, Any],
    detector_configs: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    start = extract_start_doc(run)
    image = _extract_image_array(run, image_key=image_key)
    detector_config = _detector_config_for_image_key(image_key, detector_configs)

    readback = _readback_from_tiling_config(start, config=config, detector_config=detector_config)
    detector_name = detector_config.get("name") or image_key.replace("_image", "")

    md = {
        "stitch_group_id": f"{start.get('stitch_group_id')}::{detector_name}",
        "sample_name": start.get("sample_name"),
        "stitch_tiling_mode": start.get("stitch_tiling_mode"),
        "stitch_tile_label": start.get("stitch_tile_label"),
        "stitch_tile_index": start.get("stitch_tile_index"),
        "stitch_tile_total": start.get("stitch_tile_total"),
        "detector_readback": readback,
        "mask_path": detector_config.get("mask_path") or None,
        "detector_name": detector_name,
        "image_key": image_key,
        "source_uid": start.get("uid"),
        "source_scan_id": start.get("scan_id"),
        "source_filename": start.get("filename"),
    }

    return {"image": image, "metadata": md, "mask": None}


def _strip_known_extension(filename: str) -> str:
    base = os.path.basename(str(filename))
    stem, ext = os.path.splitext(base)
    if ext.lower() in {".tif", ".tiff", ".npz", ".json", ".npy", ".edf"}:
        return stem
    return base


def _format_rule_template(template: str, values: Mapping[str, str]) -> str:
    return str(template).format(**{key: str(value) for key, value in values.items()})


def _output_rule_values(detector_name: str, mode: str) -> Dict[str, str]:
    return {
        "detector": detector_name,
        "mode": mode,
        "stitched_token": f"stitched_{mode}",
    }


def _stitched_output_base_name(group_id: str, tiles: List[Any], output_config: Mapping[str, Any] | None = None) -> str:
    outputs = output_config or {}
    filename_rule = outputs.get("filename_rule", {})
    ordered = sorted(tiles, key=lambda t: (t.metadata.stitch_tile_index, t.metadata.stitch_tile_label))
    if ordered:
        tile = ordered[-1]
        filename = tile.metadata.source_filename
        label = str(tile.metadata.stitch_tile_label)
        mode = str(tile.metadata.stitch_tiling_mode)
        if filename:
            base = _strip_known_extension(str(filename))
            values = _output_rule_values("", mode)
            stitched_token_template = filename_rule.get("stitched_token_template", "stitched_{mode}")
            values["stitched_token"] = _format_rule_template(stitched_token_template, values)
            replacement_template = filename_rule.get("replace_tile_label_with", "{stitched_token}")
            stitched_token = _format_rule_template(replacement_template, values)
            replaced = re.sub(rf"(^|_){re.escape(label)}(?=_|$)", rf"\1{stitched_token}", base, count=1)
            if replaced == base and bool(filename_rule.get("append_if_tile_label_missing", True)):
                replaced = f"{base}_{stitched_token}"
            return safe_name(replaced)
    return safe_name(group_id)


def _output_paths(
    out_dir: str,
    detector_name: str,
    mode: str,
    output_base_name: str,
    output_config: Mapping[str, Any],
) -> Dict[str, str]:
    values = _output_rule_values(detector_name, mode)
    filename_rule = output_config.get("filename_rule", {})
    stitched_token_template = filename_rule.get("stitched_token_template", "stitched_{mode}")
    values["stitched_token"] = _format_rule_template(stitched_token_template, values)
    subfolder_template = output_config.get("subfolder_template", "{detector}/{stitched_token}")
    image_format = str(output_config.get("image_format", "tiff")).lower()
    image_extension = str(output_config.get("image_extension", ".tiff"))
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    output_subfolder = _format_rule_template(subfolder_template, values)
    subfolder_parts = [safe_name(part) for part in output_subfolder.replace("\\", "/").split("/") if part]
    detector_output_dir = os.path.join(out_dir, *subfolder_parts)
    return {
        "directory": detector_output_dir,
        "image_format": image_format,
        "image": os.path.join(detector_output_dir, f"{output_base_name}{image_extension}"),
        "json": os.path.join(detector_output_dir, f"{output_base_name}.json"),
    }


def _fetch_runs(scan_ids: Iterable[int], tiled_uri: str, catalog_path: str) -> List[Any]:
    client = from_uri(tiled_uri)
    print(f"Connected to Tiled at {tiled_uri}")

    if "/" in catalog_path:
        parts = catalog_path.split("/")
        node = client
        for part in parts:
            node = node[part]
    else:
        node = client[catalog_path]

    print(f"Accessed catalog path: {catalog_path}")

    runs: List[Any] = []
    for sid in scan_ids:
        try:
            result = node.search(Key("scan_id") == int(sid))
            if len(result) < 1:
                continue

            matches = [result[key] for key in result]
            run = next(
                (candidate for candidate in matches if extract_start_doc(candidate).get("stitch_group_id")),
                matches[0],
            )
            runs.append(run)
        except Exception as exc:
            print(f"Skipping scan_id={sid}: {exc}")

    print(f"Found {len(runs)} runs by scan_id query")
    return runs


def _catalog_node(tiled_uri: str, catalog_path: str) -> Any:
    client = from_uri(tiled_uri)
    print(f"Connected to Tiled at {tiled_uri}")

    if "/" in catalog_path:
        parts = catalog_path.split("/")
        node = client
        for part in parts:
            node = node[part]
    else:
        node = client[catalog_path]

    print(f"Accessed catalog path: {catalog_path}")
    return node


def _find_run_by_scan_id(node: Any, scan_id: int) -> Any | None:
    result = node.search(Key("scan_id") == int(scan_id))
    if len(result) < 1:
        return None

    matches = [result[key] for key in result]
    return next(
        (candidate for candidate in matches if extract_start_doc(candidate).get("stitch_group_id")),
        matches[0],
    )


def _find_run_by_uid(node: Any, uid: str) -> Any | None:
    try:
        run = node[uid]
        extract_start_doc(run)
        return run
    except Exception:
        pass

    result = node.search(Key("uid") == str(uid))
    if len(result) < 1:
        return None
    key = next(iter(result))
    return result[key]


def _fetch_anchor_runs(
    tiled_uri: str,
    catalog_path: str,
    anchor_scan: int | None,
    anchor_uid: str | None,
    max_lookback: int,
) -> tuple[List[Any], List[int]]:
    node = _catalog_node(tiled_uri=tiled_uri, catalog_path=catalog_path)

    anchor_run = None
    if anchor_scan is not None:
        anchor_run = _find_run_by_scan_id(node, anchor_scan)
        if anchor_run is None:
            raise RuntimeError(f"Anchor scan_id={anchor_scan} was not found.")
    elif anchor_uid is not None:
        anchor_run = _find_run_by_uid(node, anchor_uid)
        if anchor_run is None:
            raise RuntimeError(f"Anchor uid={anchor_uid!r} was not found.")
    else:
        raise RuntimeError("Provide either anchor_scan or anchor_uid.")

    anchor_start = extract_start_doc(anchor_run)
    group_id = anchor_start.get("stitch_group_id")
    mode = anchor_start.get("stitch_tiling_mode")
    anchor_scan_id = anchor_start.get("scan_id")
    if group_id is None or mode is None or anchor_scan_id is None:
        raise RuntimeError(
            "Anchor run is missing required metadata. Need stitch_group_id, stitch_tiling_mode, and scan_id."
        )

    required_labels = get_required_labels(str(mode))
    max_lookback = max(int(max_lookback), 1)
    lower_scan = int(anchor_scan_id) - max_lookback + 1

    found_by_label: Dict[str, Any] = {}
    for scan_id in range(int(anchor_scan_id), lower_scan - 1, -1):
        run = _find_run_by_scan_id(node, scan_id)
        if run is None:
            continue

        start = extract_start_doc(run)
        if str(start.get("stitch_group_id")) != str(group_id):
            continue
        if str(start.get("stitch_tiling_mode")) != str(mode):
            continue

        tile_label = str(start.get("stitch_tile_label"))
        if tile_label in required_labels and tile_label not in found_by_label:
            found_by_label[tile_label] = run
            if len(found_by_label) == len(required_labels):
                break

    missing_labels = [label for label in required_labels if label not in found_by_label]
    if missing_labels:
        raise RuntimeError(
            f"Could not find all required tiles for mode={mode!r} from anchor scan {anchor_scan_id}. "
            f"Missing labels: {missing_labels}. Lookback window: {max_lookback} scans."
        )

    runs = list(found_by_label.values())
    runs.sort(key=lambda run: int(extract_start_doc(run).get("scan_id", 0)))
    scan_ids = [int(extract_start_doc(run).get("scan_id")) for run in runs]
    scan_range = [min(scan_ids), max(scan_ids)]

    print(
        "Resolved anchor group "
        f"{group_id!r}, mode={mode!r}, labels={required_labels}, scans={scan_ids}"
    )
    return runs, scan_range


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
) -> Dict[str, Any]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    out_dir = out_dir or os.path.join("outputs", "phase1_validation")

    config = load_config(config_path)
    detector_configs = config.get("detector", {}).get("image_streams", [])
    output_config = config.get("outputs", {})

    os.makedirs(out_dir, exist_ok=True)

    use_anchor = anchor_scan is not None or anchor_uid is not None
    if use_anchor:
        runs, resolved_scan_range = _fetch_anchor_runs(
            tiled_uri=tiled_uri,
            catalog_path=catalog_path,
            anchor_scan=anchor_scan,
            anchor_uid=anchor_uid,
            max_lookback=max_lookback,
        )
    else:
        if start_scan is None or end_scan is None:
            raise RuntimeError("Provide start_scan and end_scan, or use anchor_scan/anchor_uid.")
        scan_ids = range(start_scan, end_scan + 1)
        runs = _fetch_runs(scan_ids, tiled_uri=tiled_uri, catalog_path=catalog_path)
        resolved_scan_range = [start_scan, end_scan]

    if not runs:
        raise RuntimeError("No runs fetched from Tiled. Check catalog path, auth, and selection arguments.")

    normalized_runs: List[Dict[str, Any]] = []
    for run in runs:
        start = extract_start_doc(run)
        if not start.get("stitch_group_id"):
            continue
        for image_key in _image_keys(run, detector_configs):
            normalized_runs.append(
                _normalize_run(run, image_key=image_key, config=config, detector_configs=detector_configs)
            )

    if not normalized_runs:
        raise RuntimeError("No runs with stitch_group_id found in selected scan range.")

    normalized_runs = disambiguate_repeated_tile_groups(normalized_runs)

    grouped = build_groups_from_tiled_runs(
        normalized_runs,
        image_loader=lambda r: np.asarray(r["image"], dtype=np.float64),
    )

    index_payload = {
        "scan_range": resolved_scan_range,
        "tiled_uri": tiled_uri,
        "catalog_path": catalog_path,
        "settings": {
            "offsets": {
                "source": "tiling_configs",
            },
            "coordinate_system": config.get("coordinate_system", {}),
            "detector_image_streams": detector_configs,
            "outputs": output_config,
        },
        "group_count": len(grouped),
        "groups": [],
    }

    for group_id, tiles in grouped.items():
        detector_config = _detector_config_for_group_id(group_id, detector_configs)
        cfg = OffsetConfig(
            pixel_size_um=float(detector_config["pixel_size_um"]),
            readback_x_key=READBACK_X_KEY,
            readback_y_key=READBACK_Y_KEY,
            x_sign=int(config.get("coordinate_system", {}).get("x_sign", 1)),
            y_sign=int(config.get("coordinate_system", {}).get("y_sign", 1)),
        )
        stitched = stitch_tiles(tiles, cfg)
        serial = result_to_serializable(stitched)

        output_base_name = _stitched_output_base_name(group_id, tiles, output_config)
        mode = str(tiles[0].metadata.stitch_tiling_mode)
        output_paths = _output_paths(
            out_dir=out_dir,
            detector_name=str(detector_config.get("name") or "unknown_detector"),
            mode=mode,
            output_base_name=output_base_name,
            output_config=output_config,
        )
        os.makedirs(output_paths["directory"], exist_ok=True)
        json_path = output_paths["json"]
        image_path = output_paths["image"]
        save_result_json(stitched, json_path)
        save_result_image(stitched, image_path, output_paths["image_format"])

        index_payload["groups"].append(
            {
                "group_id": group_id,
                "detector": detector_config.get("name"),
                "pixel_size_um": detector_config.get("pixel_size_um"),
                "tile_count": len(tiles),
                "json": json_path,
                "image": image_path,
                "image_format": output_paths["image_format"],
                **({"tiff": image_path} if output_paths["image_format"] in {"tif", "tiff"} else {}),
                **({"npz": image_path} if output_paths["image_format"] == "npz" else {}),
                "output_base_name": output_base_name,
                "qa_flags": serial["result_metadata"].get("qa_flags", []),
            }
        )

        print(f"Stitched group {group_id} with {len(tiles)} tiles -> {image_path}")

    index_path = os.path.join(out_dir, "validation_index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2)

    print(f"Validation complete. Index written to {index_path}")
    return {
        "output_dir": os.path.abspath(out_dir),
        "index_path": os.path.abspath(index_path),
        "scan_range": resolved_scan_range,
        "group_count": len(grouped),
    }


def stitch_scans(
    start_scan,
    end_scan,
    out_dir,
    tiled_uri,
    catalog_path,
    config_path,
    anchor_scan=None,
    anchor_uid=None,
    max_lookback=50,
):
    print(f"\n{'='*60}")
    print(f"Stitching scans {start_scan}-{end_scan}...")
    print(f"{'='*60}")
    run_stitch_validation(
        start_scan=start_scan,
        end_scan=end_scan,
        anchor_scan=anchor_scan,
        anchor_uid=anchor_uid,
        max_lookback=max_lookback,
        tiled_uri=tiled_uri,
        catalog_path=catalog_path,
        config_path=config_path,
        out_dir=out_dir,
    )
    return True


def load_index(index_path):
    path = resolve_path(index_path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def list_groups(index_path):
    index_payload, path = load_index(index_path)
    groups = index_payload.get("groups", [])
    print(f"Index: {path}")
    print(f"Scan range: {index_payload.get('scan_range')}")
    print(f"Groups: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"  {i}: {group.get('group_id')}")
        print(f"     image: {group.get('image') or group.get('tiff') or group.get('npz')}")
    return groups


def extract_start_doc(run):
    if hasattr(run, "start"):
        try:
            return dict(run.start)
        except Exception:
            pass
    metadata = getattr(run, "metadata", {})
    if isinstance(metadata, Mapping) and "start" in metadata:
        return dict(metadata["start"])
    return {}


def detector_config_for_group(group, detector_configs):
    detector_name = group.get("detector")
    if detector_name:
        for detector_config in detector_configs:
            if detector_config.get("name") == detector_name:
                return detector_config

    group_id = str(group.get("group_id", ""))
    for detector_config in detector_configs:
        name = str(detector_config.get("name", ""))
        if name and f"::{name}::" in group_id:
            return detector_config
    return {}


def extract_image(run, detector_config=None):
    if hasattr(run, "primary"):
        primary = run.primary.read()
    else:
        primary = run["primary"]["data"].read()

    candidates = [key for key in primary.data_vars if str(key).endswith("_image")]
    if not candidates:
        candidates = list(primary.data_vars)

    if detector_config:
        image_key = str(detector_config.get("image_key"))
        candidates = [key for key in candidates if str(key) == image_key]
        if not candidates:
            raise KeyError(f"Configured image_key {image_key!r} was not found in primary stream")

    image = np.asarray(primary[candidates[0]])
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[0]
    return np.asarray(image, dtype=float)


def select_stitch_run(result):
    matches = [result[key] for key in result]
    return next(
        (candidate for candidate in matches if extract_start_doc(candidate).get("stitch_group_id")),
        matches[0],
    )


def to_log_display(image):
    return np.log10(np.clip(np.abs(np.asarray(image, dtype=float)), 1e-3, None))


def build_signal_mask(image):
    image = np.asarray(image)
    finite = np.isfinite(image)
    if not np.any(finite):
        return finite

    background_value = float(np.min(image[finite]))
    if background_value < 0:
        return finite & (image > background_value + 1e-12)
    return finite & (image != 0)


def compute_signal_bbox(image, padding=40):
    rows, cols = np.nonzero(image)
    if rows.size == 0 or cols.size == 0:
        return None

    row_min = max(int(rows.min()) - padding, 0)
    row_max = min(int(rows.max()) + padding + 1, image.shape[0])
    col_min = max(int(cols.min()) - padding, 0)
    col_max = min(int(cols.max()) + padding + 1, image.shape[1])
    return row_min, row_max, col_min, col_max


def load_stitched_image(group):
    if group.get("image") and str(group.get("image_format", "")).lower() in {"tif", "tiff"}:
        return np.asarray(Image.open(resolve_path(group["image"])), dtype=float)
    if group.get("image") and str(group.get("image_format", "")).lower() == "npz":
        return np.load(resolve_path(group["image"]))["stitched_image"]
    if group.get("tiff"):
        return np.asarray(Image.open(resolve_path(group["tiff"])), dtype=float)
    if group.get("npz"):
        return np.load(resolve_path(group["npz"]))["stitched_image"]
    raise KeyError("Group does not contain a stitched image path ('tiff' or legacy 'npz').")


def sample_name_from_group_id(group_id):
    parts = str(group_id).split("::")
    if len(parts) >= 3:
        return "::".join(parts[2:])
    return str(group_id)


def collect_source_tiles(index_payload, group):
    target_group = group["group_id"]
    detector_configs = index_payload.get("settings", {}).get("detector_image_streams", [])
    detector_config = detector_config_for_group(group, detector_configs)
    detector_name = detector_config.get("name") or group.get("detector")
    client = from_uri(index_payload["tiled_uri"])
    node = client
    for part in index_payload["catalog_path"].split("/"):
        node = node[part]

    scan_start, scan_end = index_payload["scan_range"]
    tiles = []
    for scan_id in range(scan_start, scan_end + 1):
        result = node.search(Key("scan_id") == int(scan_id))
        if len(result) < 1:
            continue

        run = select_stitch_run(result)
        start = extract_start_doc(run)
        if not start.get("stitch_group_id"):
            continue

        group_key = f"{start.get('stitch_group_id')}::{detector_name}::{start.get('sample_name') or 'unknown_sample'}"
        if not (group_key == target_group or target_group.startswith(group_key + "__rep")):
            continue

        tile_index = int(start.get("stitch_tile_index", 0))
        tile_label = start.get("stitch_tile_label", f"tile{tile_index}")
        source_scan_id = start.get("scan_id", scan_id)
        tiles.append((tile_index, tile_label, source_scan_id, extract_image(run, detector_config)))

    return sorted(tiles, key=lambda item: item[0])


def plot_group(group_index=0, index_path=FALLBACK_INDEX_PATH, save_path=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if group_index < 0 or group_index >= len(groups):
        raise ValueError(f"group index must be between 0 and {len(groups) - 1}")

    group = groups[group_index]
    target_group = group["group_id"]
    tiles = collect_source_tiles(index_payload, group)
    stitched = load_stitched_image(group)
    tile_summary = ", ".join([f"{label}[{tile_index}]" for tile_index, label, _, _ in tiles]) or "none"

    tile_displays = [to_log_display(image) for _, _, _, image in tiles]
    display_img = to_log_display(stitched)
    finite_chunks = [image[np.isfinite(image)] for image in tile_displays + [display_img]]
    finite_chunks = [chunk for chunk in finite_chunks if chunk.size > 0]
    if finite_chunks:
        global_signal = np.concatenate(finite_chunks)
        shared_vmin, shared_vmax = np.percentile(global_signal, global_percentiles)
        if shared_vmin == shared_vmax:
            shared_vmin = float(global_signal.min())
            shared_vmax = float(global_signal.max())
    else:
        shared_vmin = -3.0
        shared_vmax = 1.0

    signal_mask = build_signal_mask(stitched)
    masked_display = np.ma.masked_where(~signal_mask, display_img)
    bbox = compute_signal_bbox(signal_mask.astype(np.uint8))

    n_panels = len(tiles) + 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = list(np.atleast_1d(axes))

    for i, (tile_index, label, source_scan_id, image) in enumerate(tiles):
        axes[i].imshow(tile_displays[i], cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
        axes[i].set_title(f"Before: {label}\nscan {source_scan_id} (index {tile_index})")
        axes[i].axis("off")

    image_after = axes[-2].imshow(display_img, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    axes[-2].set_title(f"After: stitched full canvas\nlabels: {tile_summary}")
    axes[-2].axis("off")

    axes[-1].imshow(masked_display, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    if bbox is not None:
        row_min, row_max, col_min, col_max = bbox
        axes[-1].set_xlim(col_min, col_max)
        axes[-1].set_ylim(row_max, row_min)
    axes[-1].set_title(f"After: cropped signal\nlabels: {tile_summary}")
    axes[-1].axis("off")

    if show_colorbar:
        colorbar = fig.colorbar(image_after, ax=axes, shrink=0.85, pad=0.01)
        colorbar.set_label("log10(|intensity|)")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(index_abs_path), f"preview_group{group_index}_{safe_name(target_group)}.png")
    save_abs_path = resolve_path(save_path)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_abs_path, dpi=150)
    plt.close(fig)

    print(f"Group: {target_group}")
    print(f"Tiles shown: {len(tiles)}")
    print(f"Stitched shape: {stitched.shape}")
    print(f"Saved preview: {save_abs_path}")
    return save_abs_path


def plot_overview(index_path=FALLBACK_INDEX_PATH, out_dir=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if out_dir is None:
        out_dir = os.path.dirname(index_abs_path)

    if not groups:
        raise ValueError("No stitched groups found in index.")

    detectors = []
    samples = []
    images = {}
    display_values = []
    for group in groups:
        detector = str(group.get("detector") or "unknown_detector")
        sample = sample_name_from_group_id(group.get("group_id", "unknown_sample"))
        if detector not in detectors:
            detectors.append(detector)
        if sample not in samples:
            samples.append(sample)

        stitched = load_stitched_image(group)
        display = to_log_display(stitched)
        images[(sample, detector)] = display
        finite = display[np.isfinite(display)]
        if finite.size:
            display_values.append(finite)

    if display_values:
        shared_vmin, shared_vmax = np.percentile(np.concatenate(display_values), global_percentiles)
    else:
        shared_vmin, shared_vmax = -3.0, 1.0
    if shared_vmin == shared_vmax:
        shared_vmin, shared_vmax = None, None

    fig, axes = plt.subplots(
        len(samples),
        len(detectors),
        figsize=(5 * len(detectors), 4.5 * len(samples)),
        squeeze=False,
    )
    image_after = None
    for row, sample in enumerate(samples):
        for col, detector in enumerate(detectors):
            ax = axes[row][col]
            display = images.get((sample, detector))
            if display is None:
                ax.axis("off")
                ax.set_title(f"{detector}\n{sample}\nmissing")
                continue
            image_after = ax.imshow(display, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
            ax.set_title(f"{detector}\n{sample}")
            ax.axis("off")

    if show_colorbar and image_after is not None:
        colorbar = fig.colorbar(image_after, ax=axes, shrink=0.85, pad=0.01)
        colorbar.set_label("log10(|intensity|)")

    save_path = os.path.join(out_dir, "preview_all_stitched_outputs.png")
    save_abs_path = resolve_path(save_path)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_abs_path, dpi=150)
    plt.close(fig)
    print(f"Saved overview preview: {save_abs_path}")
    return save_abs_path


def plot_all(index_path=FALLBACK_INDEX_PATH, out_dir=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if out_dir is None:
        out_dir = os.path.dirname(index_abs_path)

    saved = []
    saved.append(plot_overview(index_path=index_path, out_dir=out_dir, global_percentiles=global_percentiles, show_colorbar=show_colorbar))
    for group_index, group in enumerate(groups):
        save_path = os.path.join(out_dir, f"preview_group{group_index}_{safe_name(group['group_id'])}.png")
        saved.append(
            plot_group(
                group_index=group_index,
                index_path=index_path,
                save_path=save_path,
                global_percentiles=global_percentiles,
                show_colorbar=show_colorbar,
            )
        )
    print(f"Saved {len(saved)} preview images.")
    return saved


def choose(value, default):
    return default if value is None else value


def run_pipeline(args):
    settings = load_settings(args.config)
    out_dir = choose(args.out_dir, output_dir_from_config(settings))
    index_path = os.path.join(out_dir, config_value(settings, "outputs", "index_filename", "validation_index.json"))
    tiled = settings.get("tiled", {})
    plotting = settings.get("plotting", {})
    global_percentiles = plotting.get("global_percentiles", [1.0, 99.5])

    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))

    os.makedirs(resolve_path(out_dir), exist_ok=True)
    if not stitch_scans(
        args.start_scan,
        args.end_scan,
        out_dir=out_dir,
        tiled_uri=choose(args.tiled_uri, tiled.get("uri", "https://tiled.nsls2.bnl.gov")),
        catalog_path=choose(args.catalog_path, tiled.get("catalog_path", "cms/raw")),
        config_path=args.config,
    ):
        print("Stitching failed.")
        return 1

    list_groups(index_path)
    if args.plot == "all":
        plot_all(index_path, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    elif args.plot is not None:
        plot_group(
            group_index=int(args.plot),
            index_path=index_path,
            global_percentiles=global_percentiles,
            show_colorbar=show_colorbar,
        )
    return 0


def run_anchor_pipeline(args):
    settings = load_settings(args.config)
    out_dir = choose(args.out_dir, output_dir_from_config(settings))
    index_path = os.path.join(out_dir, config_value(settings, "outputs", "index_filename", "validation_index.json"))
    tiled = settings.get("tiled", {})
    plotting = settings.get("plotting", {})
    global_percentiles = plotting.get("global_percentiles", [1.0, 99.5])

    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))

    os.makedirs(resolve_path(out_dir), exist_ok=True)
    if not stitch_scans(
        None,
        None,
        out_dir=out_dir,
        tiled_uri=choose(args.tiled_uri, tiled.get("uri", "https://tiled.nsls2.bnl.gov")),
        catalog_path=choose(args.catalog_path, tiled.get("catalog_path", "cms/raw")),
        config_path=args.config,
        anchor_scan=args.scan_id,
        anchor_uid=args.uid,
        max_lookback=args.max_lookback,
    ):
        print("Stitching failed.")
        return 1

    list_groups(index_path)
    if args.plot == "all":
        plot_all(index_path, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    elif args.plot is not None:
        plot_group(
            group_index=int(args.plot),
            index_path=index_path,
            global_percentiles=global_percentiles,
            show_colorbar=show_colorbar,
        )
    return 0


def list_command(args):
    settings = load_settings(args.config)
    index_path = choose(args.index_path, index_path_from_config(settings))
    list_groups(index_path)
    return 0


def plot_command(args):
    settings = load_settings(args.config)
    plotting = settings.get("plotting", {})
    index_path = choose(args.index_path, index_path_from_config(settings))
    global_percentiles = choose(args.global_percentiles, plotting.get("global_percentiles", [1.0, 99.5]))
    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))
    plot_group(
        group_index=args.group_index,
        index_path=index_path,
        save_path=args.save,
        global_percentiles=global_percentiles,
        show_colorbar=show_colorbar,
    )
    return 0


def plot_all_command(args):
    settings = load_settings(args.config)
    plotting = settings.get("plotting", {})
    index_path = choose(args.index_path, index_path_from_config(settings))
    global_percentiles = choose(args.global_percentiles, plotting.get("global_percentiles", [1.0, 99.5]))
    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))
    plot_all(index_path, out_dir=args.out_dir, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Single entry point for stitching and plotting Tiled scan ranges.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="stitch a scan range")
    run_parser.add_argument("start_scan", type=int)
    run_parser.add_argument("end_scan", type=int)
    run_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument(
        "--plot",
        nargs="?",
        const="all",
        default=None,
        help="plot all groups; optionally pass one group index, or use 'all' explicitly",
    )
    run_parser.add_argument("--show-colorbar", action="store_true", default=None)
    run_parser.add_argument("--out-dir", default=None)
    run_parser.add_argument("--tiled-uri", default=None)
    run_parser.add_argument("--catalog-path", default=None)
    run_parser.set_defaults(func=run_pipeline)

    run_anchor_parser = subparsers.add_parser(
        "run-anchor",
        help="stitch one acquisition by searching backward from a scan_id or uid",
    )
    anchor_group = run_anchor_parser.add_mutually_exclusive_group(required=True)
    anchor_group.add_argument("--scan-id", type=int, default=None)
    anchor_group.add_argument("--uid", default=None)
    run_anchor_parser.add_argument("--max-lookback", type=int, default=50)
    run_anchor_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run_anchor_parser.add_argument(
        "--plot",
        nargs="?",
        const="all",
        default=None,
        help="plot all groups; optionally pass one group index, or use 'all' explicitly",
    )
    run_anchor_parser.add_argument("--show-colorbar", action="store_true", default=None)
    run_anchor_parser.add_argument("--out-dir", default=None)
    run_anchor_parser.add_argument("--tiled-uri", default=None)
    run_anchor_parser.add_argument("--catalog-path", default=None)
    run_anchor_parser.set_defaults(func=run_anchor_pipeline)

    list_parser = subparsers.add_parser("list", help="list groups in a validation index")
    list_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    list_parser.add_argument("--index-path", default=None)
    list_parser.set_defaults(func=list_command)

    plot_parser = subparsers.add_parser("plot", help="plot one stitched output group")
    plot_parser.add_argument("group_index", type=int, nargs="?", default=0)
    plot_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    plot_parser.add_argument("--index-path", default=None)
    plot_parser.add_argument("--save", default=None)
    plot_parser.add_argument("--show-colorbar", action="store_true", default=None)
    plot_parser.add_argument("--global-percentiles", nargs=2, type=float, default=None)
    plot_parser.set_defaults(func=plot_command)

    plot_all_parser = subparsers.add_parser("plot-all", help="plot every group in a validation index")
    plot_all_parser.add_argument("--cionfig", default=DEFAULT_CONFIG_PATH)
    plot_all_parser.add_argument("--index-path", default=None)
    plot_all_parser.add_argument("--out-dir", default=None)
    plot_all_parser.add_argument("--show-colorbar", action="store_true", default=None)
    plot_all_parser.add_argument("--global-percentiles", nargs=2, type=float, default=None)
    plot_all_parser.set_defaults(func=plot_all_command)

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0].isdigit() and argv[1].isdigit():
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
