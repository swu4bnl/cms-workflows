import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from tiled.client import from_uri
from tiled.queries import Key

from .core.adapters.tiled_adapter import build_groups_from_tiled_runs
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
    """Resolve all tiles for one stitch group by walking backward from an anchor run.

    The anchor run identifies the target ``stitch_group_id`` and
    ``stitch_tiling_mode``. The required tile labels come from the configured
    tiling mode, and each label must be found within ``max_lookback`` scan IDs.
    """
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
