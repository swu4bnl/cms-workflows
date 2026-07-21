from prefect import get_run_logger, task


@task(retries=2, retry_delay_seconds=10)
def run_auto_stitch_anchor(uid, api_key=None, stitch_config=None):
    """Run the anchor auto-stitching for a given run.

    Parameters
    ----------
    uid : str
        The UID of the run to stitch.
    api_key : str, optional
        API key for Tiled access.
    stitch_config : dict, optional
        Additional configuration for the stitching operation.
    """
    logger = get_run_logger()
    logger.info(f"Running anchor auto-stitch for uid {uid}")
    if stitch_config:
        logger.info(f"Stitch config: {stitch_config}")
    # TODO: implement actual stitching logic
    logger.info(f"Anchor auto-stitch complete for uid {uid}")


@task(retries=2, retry_delay_seconds=10)
def verify_stitch_outputs(uid, api_key=None):
    """Verify the outputs of the stitching operation.

    Parameters
    ----------
    uid : str
        The UID of the run to verify.
    api_key : str, optional
        API key for Tiled access.
    """
    logger = get_run_logger()
    logger.info(f"Verifying stitch outputs for uid {uid}")
    # TODO: implement actual output verification
    logger.info(f"Stitch output verification complete for uid {uid}")
