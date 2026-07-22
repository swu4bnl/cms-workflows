from prefect import task, get_run_logger
from pathlib import Path
import os
import glob
from data_validation import get_run


PROPOSAL_ROOT = Path("/nsls2/data/cms/proposals")


def detector_mapping(detector):
    if detector in {"pilatus300k-1", "pilatus800k-2"}:
        return "maxs"
    elif detector == "pilatus2m-1":
        return "saxs"
    elif detector == "pilatus800k-1":
        return "waxs"
    elif "webcam" in detector:
        return detector
    else:
        return None

def chmod_and_chown(path, *, uid=None, gid=None, mode=0o775):
    os.chmod(path, mode)

    # The following needs to be tested more; Prefect worker account doesn't have permissions to chown
    #if uid is not None and gid is not None:
    #    os.chown(path, uid, gid)

def make_relative_path(path_value):
    """Prevent absolute metadata paths from escaping the proposal directory."""
    path = Path(path_value)
    return Path(*path.parts[1:]) if path.is_absolute() else path

def proposal_directory(doc):
    """Return the proposal directory for a run start document."""
    return PROPOSAL_ROOT / doc["cycle"] / doc["data_session"]

def experiment_directory(doc):
    """Return the experiments directory inside the proposal directory."""
    return proposal_directory(doc) / make_relative_path(doc.get("experiments_directory", "experiments"))

def experiment_alias_directory(doc):
    """Return the user-facing experiment alias directory, or ``None`` if unset."""
    path_expr_alias = doc.get("experiment_alias_directory")
    if not path_expr_alias:
        return None
    return experiment_directory(doc) / make_relative_path(path_expr_alias)

@task(retries=2, retry_delay_seconds=10)
def create_symlinks(ref, api_key=None, dry_run=False):
    """
    Parameters
    ----------
    ref : Union[int, str]
        Scan_id or uid of the start document

    """
    logger = get_run_logger()

    hrf = get_run(ref, api_key=api_key)
    for name, doc in hrf.documents():
        if name == "start":
            if detectors := doc.get("detectors"):
                pass
            else:
                logger.info("Not a measurement scan")
                return
            if filename := doc.get("filename"):
                pass
            else:
                logger.info("Skipping the creation of the link because 'filename' is not set.")
                return
            path_expr_alias = experiment_alias_directory(doc)
            if path_expr_alias:
                # stats = path_proposal.stat()
                path_expr = experiment_directory(doc)
                if dry_run:
                    logger.info(f"Dry run: mkdir {path_expr}")
                else:
                    path_expr.mkdir(exist_ok=True, parents=True)
                #chmod_and_chown(path_expr, uid=stats.st_uid, gid=stats.st_gid)
                if dry_run:
                    logger.info(f"Dry run: mkdir {path_expr_alias}")
                else:
                    path_expr_alias.mkdir(exist_ok=True, parents=True)
                #chmod_and_chown(path_expr_alias, uid=stats.st_uid, gid=stats.st_gid)
            else:
                logger.info("Directory for links is not specified; skipping.")
                return

        elif name == "resource":
            for det in detectors:
                if det in doc["root"]:
                    if detname := detector_mapping(det):
                        # Define subfolders for "raw" and "analysis", but not for cameras
                        subdir_raw = "camera" if "webcam" in detname else f"{detname}/raw"
                        subdir_analysis = "camera" if "webcam" in detname else f"{detname}/analysis"
                        path_analysis = Path(path_expr_alias) / subdir_analysis
                        if dry_run:
                            logger.info(f"Dry run: mkdir {path_analysis}")
                        else:
                            path_analysis.mkdir(exist_ok=True, parents=True)
                        # chmod_and_chown(path_analysis, uid=stats.st_uid, gid=stats.st_gid)
                        # chmod_and_chown(path_analysis.parent, uid=stats.st_uid, gid=stats.st_gid)
                        path_data = Path(path_expr_alias) / 'data'
                        if dry_run:
                            logger.info(f"Dry run: mkdir {path_data}")
                        else:
                            path_data.mkdir(exist_ok=True, parents=True)
                        # chmod_and_chown(path_data, uid=stats.st_uid, gid=stats.st_gid)
                        logger.info(f"Created analysis and data folders for {det}")

                        if 'TIFF' in doc['spec']:
                            prefix = str(Path(doc["root"]) / doc["resource_path"] / doc["resource_kwargs"]["filename"])
                            ext = doc["resource_kwargs"]["template"].split('.')[-1]
                        elif 'HDF5' in doc['spec']:
                            prefix = str(Path(doc["root"]) / doc["resource_path"])
                            ext = doc["resource_path"].split('.')[-1]
                        else:
                            logger.info(f"The output for this spec has not been implemented yet. {doc['spec']}")
                            return
    
                        for file_path in glob.glob(prefix + "*"):
                            source_name = os.path.splitext(os.path.basename(file_path))[0]  # only file name w/o extension
                            name, indx = source_name.split("_")    # filename and index of the image
                            link_path = Path(path_expr_alias) / subdir_raw / f"{filename or name}_{indx}_{detname}.{ext}"
                            if link_path.exists():
                                logger.info("Scan was run already")
                            else:
                                if dry_run:
                                    logger.info(f"Dry run: mkdir {link_path}")
                                    logger.info(f"Dry run: Linked: {file_path} to {link_path}")
                                else:
                                    link_path.parent.mkdir(exist_ok=True, parents=True)
                                    # chmod_and_chown(link_path.parent, uid=stats.st_uid, gid=stats.st_gid)
                                    os.symlink(file_path, link_path)
                                    logger.info(f"Linked: {file_path} to {link_path}")
                        break
            else:
                logger.error(f"Resource document referencing unknown detector {det}.")
