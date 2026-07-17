import os
import subprocess
import traceback
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.notifications import SlackWebhook
from prefect.context import FlowRunContext
from prefect.settings import PREFECT_UI_URL

#from analysis import run_analysis
from data_validation import data_validation_task, get_run
from dev_probe import test as dev_probe_test
from linker import create_symlinks
from dotenv import load_dotenv

CATALOG_NAME = "cms"


def slack(func):
    """
    Send a message to mon-prefect and mon-prefect-cs slack channels if the flow-run failed.
    Send a message to mon-prefect-cms slack channel with the flow-run status.
    Send a message to mon-bluesky slack channel if the bluesky-run failed.

    NOTE: the name of this inner function is the same as the real end_of_workflow() function because
    when the decorator is used, Prefect sees the name of this inner function as the name of
    the flow. To keep the naming of workflows consistent, the name of this inner function had to match the expected name.
    """

    def end_of_run_workflow(stop_doc, api_key=None, dry_run=False, **kwargs):
        flow_run_name = FlowRunContext.get().flow_run.dict().get("name")

        # Load slack credentials that are saved in Prefect.
        mon_prefect = SlackWebhook.load("mon-prefect")
        mon_bluesky = SlackWebhook.load("mon-bluesky")
        mon_prefect_cms = SlackWebhook.load("mon-prefect-cms")
        mon_prefect_cs = SlackWebhook.load("mon-prefect-cs")

        # Get the uid.
        uid = stop_doc["run_start"]

        # Get the scan_id.
        run = get_run(uid, api_key=api_key)
        scan_id = run.start["scan_id"]

        # Send a message to mon-bluesky if bluesky-run failed.
        if stop_doc.get("exit_status") == "fail":
            mon_bluesky.notify(
                f":bangbang: {CATALOG_NAME} bluesky-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```reason: {stop_doc.get('reason', 'none')}```"
            )

        try:
            result = func(stop_doc, api_key=api_key, dry_run=dry_run, **kwargs)

            # Send a message to mon-prefect-cms if flow-run is successful.
            message = f":white_check_mark: {CATALOG_NAME} flow-run successful. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}```"
            mon_prefect_cms.notify(message)
            return result
        except Exception as e:
            tb = traceback.format_exception_only(e)

            # Send a message to mon-prefect-cms, mon-prefect if flow-run failed.
            message = f":bangbang: {CATALOG_NAME} flow-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```{tb[-1]}```"
            mon_prefect.notify(message)
            mon_prefect_cms.notify(message)
            flow_run = FlowRunContext.get().flow_run
            # Add link to flow-run for the message to mon-prefect-cs.
            program_message = (
                f":bangbang: {CATALOG_NAME} flow-run failed. <{PREFECT_UI_URL.value()}/flow-runs/"
                + f"flow-run/{flow_run.id}|the flow run link> (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```{tb[-1]}```"
            )
            mon_prefect_cs.notify(program_message)
            raise

    return end_of_run_workflow


@task
def log_completion():
    logger = get_run_logger()
    logger.info("Complete")


def _categorize_anchor_failure(stderr: str) -> str:
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
def verify_stitch_outputs(stitch_result):
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
        "Output verification passed. validation_index=%s tiff=%s sidecar_json=%s preview_png=%s uid=%s gid=%s",
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


@task
def run_auto_stitch_anchor(
    uid,
    api_key=None,
    stitch_repo_path=None,
    max_lookback=50,
    config_path="configs/stitching_defaults.json",
    out_dir="stitching_outputs_prefect",
    plot=True,
    show_colorbar=True,
    tiled_uri=None,
    catalog_path=None,
):
    """Run Part 2 stitching CLI in anchor mode using scan_id resolved from run UID."""
    logger = get_run_logger()
    run = get_run(uid, api_key=api_key)
    scan_id = int(run.start["scan_id"])

    bundled_repo = Path(__file__).resolve().parent / "auto_stitch"

    repo_root = Path(
        stitch_repo_path
        or os.getenv("CMS_AUTO_STITCH_REPO", "")
        or str(bundled_repo)
    ).expanduser()
    if not repo_root.exists() or not (repo_root / "stitch.py").exists():
        raise FileNotFoundError(
            "Auto-stitch repo is not available for anchor mode. "
            f"Expected stitch.py under: {repo_root}"
        )

    command = [
        "pixi",
        "run",
        "python",
        "stitch.py",
        "run-anchor",
        "--scan-id",
        str(scan_id),
        "--max-lookback",
        str(int(max_lookback)),
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
    ]
    if plot:
        command.append("--plot")
    if show_colorbar:
        command.append("--show-colorbar")
    if tiled_uri:
        command.extend(["--tiled-uri", str(tiled_uri)])
    if catalog_path:
        command.extend(["--catalog-path", str(catalog_path)])

    logger.info(
        "Launching anchor-mode auto-stitch for uid=%s scan_id=%s in repo=%s",
        uid,
        scan_id,
        str(repo_root),
    )
    proc = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        logger.info(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            logger.error(proc.stderr)
        category = _categorize_anchor_failure(proc.stderr)
        raise RuntimeError(
            "Anchor auto-stitch failed "
            f"for scan_id={scan_id} (exit={proc.returncode}, category={category})"
        )

    logger.info("Anchor auto-stitch completed for scan_id=%s", scan_id)
    output_dir = (repo_root / out_dir).resolve()
    return {
        "uid": uid,
        "scan_id": scan_id,
        "out_dir": str(out_dir),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
    }


@flow(task_runner=ConcurrentTaskRunner())
@slack
def end_of_run_workflow(
    stop_doc,
    api_key=None,
    dry_run=False,
    enable_anchor_autostitch=False,
    stitch_repo_path=None,
    max_lookback=50,
    stitch_config_path="configs/stitching_defaults.json",
    stitch_out_dir="stitching_outputs_prefect",
    stitch_plot=True,
    stitch_show_colorbar=True,
    stitch_tiled_uri=None,
    stitch_catalog_path=None,
    verify_anchor_outputs=True,
):
    load_dotenv()
    logger = get_run_logger()
    uid = stop_doc["run_start"]

    # Dev-only probe for branch/deployment validation.
    dev_probe_test(uid)
    logger.info("Dev probe module executed")

    # Launch validation, analysis, and linker tasks concurrently
    linker_task = create_symlinks.submit(uid, api_key=api_key, dry_run=dry_run)
    logger.info("Launched linker task")

    validation_task = data_validation_task.submit(uid, api_key=api_key)
    logger.info("Launched validation tasks")

    stitch_task = None
    if bool(enable_anchor_autostitch):
        stitch_task = run_auto_stitch_anchor.submit(
            uid,
            api_key=api_key,
            stitch_repo_path=stitch_repo_path,
            max_lookback=max_lookback,
            config_path=stitch_config_path,
            out_dir=stitch_out_dir,
            plot=stitch_plot,
            show_colorbar=stitch_show_colorbar,
            tiled_uri=stitch_tiled_uri,
            catalog_path=stitch_catalog_path,
        )
        logger.info("Launched anchor auto-stitch task")
    else:
        logger.info("Anchor auto-stitch is disabled for this deployment")

    # analysis_task = run_analysis(raw_ref=uid)
    # logger.info("Launched analysis task")

    # Wait for all tasks to comple
    logger.info("Waiting for tasks to complete")
    linker_task.result()
    validation_task.result()
    if stitch_task is not None:
        stitch_result = stitch_task.result()
        if bool(verify_anchor_outputs):
            verify_stitch_outputs.submit(stitch_result).result()
    # analysis_task.result()
    log_completion()
