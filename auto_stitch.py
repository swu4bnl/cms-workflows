"""Prefect tasks for anchor-mode auto-stitching integration.

Follows the same pattern as linker.py and data_validation.py: one module
per concern, imported by end_of_run_workflow.py for orchestration.
"""

import os
import importlib.util
import sys
from pathlib import Path

from prefect import get_run_logger, task

from data_validation import get_run


def _bundled_repo_path() -> Path:
    return Path(__file__).resolve().parent / "auto_stitch"


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_stitch_runner(repo_root: Path):
    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    module_path = repo_root / "stitch.py"
    spec = importlib.util.spec_from_file_location("cms_workflows_stitch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load stitch runner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_stitch_validation


def _categorize_anchor_failure(stderr: str) -> str:
    """Map stitch CLI stderr to a short human-readable failure category."""
    text = (stderr or "").lower()
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
    """Run anchor-mode auto-stitch via the bundled stitch.py CLI adapter.

    Parameters
    ----------
    uid : str
        Run UID used to resolve the scan_id from Tiled.
    api_key : str, optional
        Tiled API key.
    stitch_config : dict, optional
        Optional overrides for stitch behavior. Recognized keys:

        - ``repo_path``: path to the stitch repo root (default: bundled
          ``auto_stitch/`` directory; env override: ``CMS_AUTO_STITCH_REPO``).
        - ``max_lookback``: scan lookback window size (default: 50).
        - ``config_path``: path to stitching config JSON relative to repo root
          (default: ``configs/stitching_defaults.json``).
        - ``out_dir``: output directory relative to repo root
          (default: ``stitching_outputs_prefect``).
        - ``tiled_uri``: Tiled server URI override.
        - ``catalog_path``: Tiled catalog path override.

    Returns
    -------
    dict
        ``{"uid": ..., "scan_id": ..., "output_dir": <absolute path>}``
    """
    logger = get_run_logger()
    config = stitch_config or {}

    run = get_run(uid, api_key=api_key)
    scan_id = int(run.start["scan_id"])

    bundled_repo = _bundled_repo_path()
    repo_root = Path(
        config.get("repo_path") or os.getenv("CMS_AUTO_STITCH_REPO", "") or str(bundled_repo)
    ).expanduser()

    if not repo_root.exists() or not (repo_root / "stitch.py").exists():
        raise FileNotFoundError(
            "Auto-stitch repo is not available. "
            f"Expected stitch.py under: {repo_root}"
        )

    max_lookback = int(config.get("max_lookback", 50))
    config_path = _resolve_repo_path(repo_root, config.get("config_path", "configs/stitching_defaults.json"))
    out_dir = _resolve_repo_path(repo_root, config.get("out_dir", "stitching_outputs_prefect"))
    tiled_uri = config.get("tiled_uri")
    catalog_path = config.get("catalog_path")

    logger.info(
        "Launching anchor-mode auto-stitch for uid=%s scan_id=%s repo=%s",
        uid,
        scan_id,
        str(repo_root),
    )

    run_stitch_validation = _load_stitch_runner(repo_root)
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
        raise FileNotFoundError(f"No preview PNGs found under {output_dir}")

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
