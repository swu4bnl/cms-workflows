import os
import time

from prefect import flow, get_run_logger, task

from tiled.client import from_uri
from dotenv import load_dotenv


def get_api_key_from_env(api_key=None):
    with open("/srv/container.secret", "r") as secrets:
        load_dotenv(stream=secrets)
    api_key = os.environ["TILED_API_KEY"]
    return api_key


@task(retries=2, retry_delay_seconds=10)
def get_run(uid, api_key=None):
    if not api_key:
        api_key = get_api_key_from_env()
    tiled_client = from_uri("https://tiled.nsls2.bnl.gov", api_key=api_key)
    run = tiled_client["cms/raw"][uid]
    return run


@task(retries=2, retry_delay_seconds=10)
def get_run_migration(uid, api_key=None): # TODO remove after migration is complete and only raw is available
    if not api_key:
        api_key = get_api_key_from_env()
    tiled_client = from_uri("https://tiled.nsls2.bnl.gov", api_key=api_key)
    run = tiled_client["cms/migration"][uid]
    return run


@task(retries=3, retry_delay_seconds=20)
def data_validation_task(uid, api_key=None):
    """Task to validate the data structure and accessibility in Tiled

    Parameters
    ----------
        uid : str
            The UID of the run to validate
        beamline_acronym : str, optional
            The acronym of the beamline (default is "cms")
    """

    logger = get_run_logger()
    logger.info("Connecting to Tiled client for beamline cms")
    run = get_run_migration(uid, api_key=api_key)
    logger.info(f"Validating uid {uid}")
    start_time = time.monotonic()
    run.validate(fix_errors=True, try_reading=True, raise_on_error=True)
    elapsed_time = time.monotonic() - start_time
    logger.info(f"Finished validating data; {elapsed_time = }")


@flow(log_prints=True)
def data_validation_flow(uid, api_key=None):
    data_validation_task(uid, api_key=api_key)
