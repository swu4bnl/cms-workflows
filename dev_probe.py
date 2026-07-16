from prefect import get_run_logger


def test(uid: str) -> None:
    """Simple dev hook to confirm new module wiring during deployment tests."""
    logger = get_run_logger()
    logger.info("dev_probe.test called with uid=%s", uid)
