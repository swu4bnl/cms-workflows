import os
from pathlib import Path
from shutil import copy2

import event_model
from prefect import flow, get_run_logger, task

from data_validation import get_run


@task
def copy_file(source, dest_dir, dry_run=False):
    logger = get_run_logger()
    if dry_run:
        logger.info(f"Dry run: copy {source} to {dest_dir}")
        return
    logger.info(f"Copying {source} to {dest_dir}")
    copy2(source, dest_dir)
    logger.info("Done copying file")


def get_det_file_paths(run):
    docs = run.documents()
    root_map = {}
    target_keys = set()
    resource_info = {}
    datum_info = {}
    source_paths = []

    for name, doc in docs:
        if name == "start":
            pass
        elif name == "resource":
            if doc["spec"] != "AD_TIFF":
                continue
            doc_root = doc["root"]
            resource_info[doc["uid"]] = {
                "path": Path(root_map.get(doc_root, doc_root)) / doc["resource_path"],
                "kwargs": doc["resource_kwargs"],
            }
        elif "datum" in name:
            if name == "datum":
                doc = event_model.pack_datum_page(doc)

            for datum_uid, point_number in zip(
                doc["datum_id"], doc["datum_kwargs"]["point_number"]
            ):
                datum_info[datum_uid] = (
                    resource_info[doc["resource"]],
                    point_number,
                )
        elif name == "descriptor":
            for key, value in doc["data_keys"].items():
                if "external" in value:
                    target_keys.add(key)
        elif "event" in name:
            if name == "event":
                doc = event_model.pack_event_page(doc)
            for key in target_keys:
                if key not in doc["data"]:
                    continue
                for datum_id in doc["data"][key]:
                    resource_vals, point_number = datum_info[datum_id]
                    orig_template = resource_vals["kwargs"]["template"]
                    frames_per_point = resource_vals["kwargs"]["frame_per_point"]

                    base_fname = resource_vals["kwargs"]["filename"]
                    for frame in range(frames_per_point):
                        source_path = Path(
                            orig_template
                            % (
                                str(resource_vals["path"]) + "/",
                                base_fname,
                                point_number * frames_per_point + frame,
                            )
                        )
                        source_paths.append(source_path)
    return source_paths


def get_data_filename(detector, dest_dir, savename, subdirs=True, dry_run=False):
    logger = get_run_logger()
    detector_map = {
        "pilatus300": "maxs",
        "pilatus300k-1": "maxs",
        "pilatus8002": "maxs",
        "pilatus800k-2": "maxs",
        "pilatus2M": "saxs",
        "pilatus2m-1": "saxs",
        "pilatus800": "waxs",
        "pilatus800k-1": "waxs",
    }
    detname = detector_map.get(detector)
    if detname is None:
        logger.warning(f"Can't do file handling for detector '{detector}'.")
        return None

    subdir = f"{detname}/raw" if subdirs else ""
    dest_path = Path(dest_dir) / subdir
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
    return dest_path / f"{savename}_{detname}.tiff"


@flow
def export(ref, api_key=None, subdirs=True, dry_run=False):
    logger = get_run_logger()
    run = get_run(ref, api_key=api_key)
    full_uid = run.start["uid"]
    logger.info(f"{full_uid = }")

    cycle = run.start.get("experiment_cycle") or run.start.get("cycle")
    if cycle is not None:
        cycle = cycle.replace("_", "-")
    logger.info(f"{cycle = }")

    proposal_num = run.start.get("experiment_proposal_number")
    logger.info(f"{proposal_num = }")
    dest_dir = f"/nsls2/data/cms/proposals/{cycle}/pass-{proposal_num}/"
    if not os.path.exists(dest_dir):
        logger.info(f"Directory {dest_dir} doesn't exist. Not copying files for {full_uid}.")
        return

    dets = run.start.get("detectors")
    savename = run.start.get("filename")
    if savename is None:
        logger.info(f"Couldn't get 'savename'. Not copying files for {full_uid}.")
        return

    resource_paths = get_det_file_paths(run)
    for detector, source_path in zip(dets, resource_paths):
        source = str(source_path)
        if not os.path.exists(source):
            logger.info(f"{source} doesn't exist. Not copying files for {full_uid}.")
            return
        dest = get_data_filename(
            detector, dest_dir, savename, subdirs=subdirs, dry_run=dry_run
        )
        if dest is None:
            continue
        copy_file(source, dest, dry_run=dry_run)
