import os
import sys
import types
import unittest
from unittest.mock import MagicMock


prefect_stub = types.ModuleType("prefect")


def task(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def get_run_logger():
    return MagicMock()


prefect_stub.task = task
prefect_stub.get_run_logger = get_run_logger
sys.modules.setdefault("prefect", prefect_stub)


import auto_stitch  # noqa: E402


class RunAutoStitchAnchorTests(unittest.TestCase):
    def test_accepts_uid_only(self):
        auto_stitch.run_auto_stitch_anchor("test-uid-123")

    def test_accepts_api_key(self):
        auto_stitch.run_auto_stitch_anchor("test-uid-123", api_key="my-key")

    def test_accepts_stitch_config(self):
        auto_stitch.run_auto_stitch_anchor(
            "test-uid-123",
            api_key="my-key",
            stitch_config={"mode": "saxs"},
        )

    def test_accepts_none_stitch_config(self):
        auto_stitch.run_auto_stitch_anchor("test-uid-123", stitch_config=None)


class VerifyStitchOutputsTests(unittest.TestCase):
    def test_accepts_uid_only(self):
        auto_stitch.verify_stitch_outputs("test-uid-123")

    def test_accepts_api_key(self):
        auto_stitch.verify_stitch_outputs("test-uid-123", api_key="my-key")


if __name__ == "__main__":
    unittest.main()
