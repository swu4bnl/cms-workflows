"""Prefect tasks for anchor-mode auto-stitching integration.

This module is the workflow-facing adapter. It resolves the completed run,
chooses the output location used by CMS proposal folders, and calls the core
stitching code in ``stitch.runner``.

Anchor mode means the Prefect task receives one completed run and treats it as
the newest tile in a stitch group. The stitch runner then looks backward through
recent scan IDs to find the other required tile labels for the same
``stitch_group_id`` and ``stitch_tiling_mode``.
"""

import os
from pathlib import Path

from prefect import get_run_logger, task

from data_validation import get_run
from linker import experiment_alias_directory


STITCH_PACKAGE_DIR = Path(__file__).resolve().parent / "stitch"
DEFAULT_STITCH_CONFIG = STITCH_PACKAGE_DIR / "configs" / "stitching_defaults.json"


def _resolve_stitch_path(path_value, default: Path) -> Path:
    """Return an absolute path, interpreting relative overrides under ``stitch/``."""
    path_value = default if path_value in (None, "") else path_value
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else STITCH_PACKAGE_DIR / path


def _default_stitch_output_dir(start_doc) -> Path:
    """Return the proposal experiment alias directory used as the stitch output base."""
    output_dir = experiment_alias_directory(start_doc)
    if output_dir is None:
        raise RuntimeError("Start document is missing experiment_alias_directory; cannot place stitch outputs.")
    return output_dir


def _categorize_anchor_failure(error_text: str) -> str:
    """Map stitch errors to a short human-readable failure category."""
    text = (error_text or "").lower()
    if "missing scan range" in text or "start_scan" in text or "end_scan" in text:
        return "missing scan range"
    if "required tiles" in text or "missing required tiles" in text or "could not find all required tiles" in text:
        return "incomplete groups"
    if "image_key" in text and "not found" in text:
        return "missing detector image key"
    if "tiled" in text and ("connect" in text or "auth" in text or "catalog" in text or "not found" in text):
        return "Tiled access failure"
    if "permission" in text or "access is denied" in text or "operation not permitted" in text:
        return "output permission failure"
    if "unsupported" in text or "configuration" in text:
        return "unsupported config"
    return "unknown failure"


@task
def run_auto_stitch_anchor(uid, api_key=None, stitch_config=None):
    """Run auto-stitch for a completed Bluesky run.

    The selected run is the anchor. Its ``scan_id`` is passed to
    ``run_stitch_validation(anchor_scan=...)`` so the stitch runner can find the
    matching tile group in Tiled. Output is written under the same proposal
    experiment alias directory used by ``linker.create_symlinks``; the stitch
    config then adds detector and stitch-mode subfolders such as
    ``maxs/stitched_ygaps``.

    ``stitch_config`` may override ``max_lookback``, ``config_path``,
    ``out_dir``, ``tiled_uri``, and ``catalog_path``. Relative config paths and
    output-directory overrides are interpreted under ``stitch/``.

    Returns ``{"uid": ..., "scan_id": ..., "output_dir": <absolute path>}``.
    """
    from stitch.runner import run_stitch_validation

    logger = get_run_logger()
    config = stitch_config or {}

    run = get_run(uid, api_key=api_key)
    scan_id = int(run.start["scan_id"])

    max_lookback = int(config.get("max_lookback", 500))
    config_path = _resolve_stitch_path(config.get("config_path"), DEFAULT_STITCH_CONFIG)
    out_dir = _resolve_stitch_path(config.get("out_dir"), _default_stitch_output_dir(run.start))
    tiled_uri = config.get("tiled_uri")
    catalog_path = config.get("catalog_path")

    logger.info(
        "Launching anchor-mode auto-stitch for uid=%s scan_id=%s output_dir=%s",
        uid,
        scan_id,
        str(out_dir),
    )

    try:
        result = run_stitch_validation(
            anchor_scan=scan_id,
            max_lookback=max_lookback,
            tiled_uri=tiled_uri or "https://tiled.nsls2.bnl.gov",
            catalog_path=catalog_path or "cms/raw",
            config_path=str(config_path),
            out_dir=str(out_dir),
        )
    except Exception as exc:
        category = _categorize_anchor_failure(str(exc))
        raise RuntimeError(
            f"Anchor auto-stitch failed for scan_id={scan_id} "
            f"(category={category})"
        ) from exc

    logger.info("Anchor auto-stitch completed for scan_id=%s", scan_id)
    return {
        "uid": uid,
        "scan_id": scan_id,
        "output_dir": result["output_dir"],
    }


@task
def verify_stitch_outputs(stitch_result):
    """Verify that expected stitch output artifacts exist and are accessible.

    Parameters
    ----------
    stitch_result : dict
        Return value from ``run_auto_stitch_anchor``.  Must contain
        ``"output_dir"``.

    Returns
    -------
    dict
        Summary of verified artifacts.
    """
    logger = get_run_logger()
    output_dir = Path(stitch_result["output_dir"]).resolve()

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    validation_index = output_dir / "validation_index.json"
    if not validation_index.exists():
        raise FileNotFoundError(f"Missing validation index: {validation_index}")

    tiff_files = sorted(list(output_dir.rglob("*.tif")) + list(output_dir.rglob("*.tiff")))
    if not tiff_files:
        raise FileNotFoundError(f"No stitched TIFF files found under {output_dir}")

    sidecar_json = [p for p in output_dir.rglob("*.json") if p.name != "validation_index.json"]
    if not sidecar_json:
        raise FileNotFoundError(f"No JSON sidecars found under {output_dir}")

preview_png = [p for p in output_dir.rglob("*.png") if "preview" in p.name.lower()]
if not preview_png:
    logger.warning("No preview PNGs found under %s", str(output_dir))

    can_read = os.access(output_dir, os.R_OK)
    can_write = os.access(output_dir, os.W_OK)
    if not (can_read and can_write):
        raise PermissionError(
            f"Output permission failure at {output_dir}: read={can_read} write={can_write}"
        )

    st = output_dir.stat()
    logger.info(
        "Output verification passed. validation_index=%s tiff=%s sidecar_json=%s "
        "preview_png=%s uid=%s gid=%s",
        str(validation_index),
        len(tiff_files),
        len(sidecar_json),
        len(preview_png),
        getattr(st, "st_uid", "n/a"),
        getattr(st, "st_gid", "n/a"),
    )

    return {
        "validation_index": str(validation_index),
        "tiff_count": len(tiff_files),
        "sidecar_json_count": len(sidecar_json),
        "preview_png_count": len(preview_png),
        "uid": getattr(st, "st_uid", None),
        "gid": getattr(st, "st_gid", None),
    }
