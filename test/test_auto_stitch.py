import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Minimal stubs so auto_stitch.py can be imported without Prefect or Tiled
# ---------------------------------------------------------------------------

prefect_stub = types.ModuleType("prefect")


def task(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


def get_run_logger():
    return None


prefect_stub.task = task
prefect_stub.get_run_logger = get_run_logger
sys.modules.setdefault("prefect", prefect_stub)

data_validation_stub = types.ModuleType("data_validation")
data_validation_stub.get_run = lambda *args, **kwargs: None
sys.modules.setdefault("data_validation", data_validation_stub)


import auto_stitch  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CategorizAnchorFailureTests(unittest.TestCase):
    def test_missing_scan_range(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("missing scan range in args"),
            "missing scan range",
        )

    def test_start_scan_keyword(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("required arg: start_scan not provided"),
            "missing scan range",
        )

    def test_end_scan_keyword(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("end_scan must be specified"),
            "missing scan range",
        )

    def test_incomplete_groups(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("could not find all required tiles for mode=ygaps"),
            "incomplete groups",
        )

    def test_missing_required_tiles(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("missing required tiles: ['pos2']"),
            "incomplete groups",
        )

    def test_missing_detector_image_key(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("image_key 'pilatus2m-1_image' not found in primary stream"),
            "missing detector image key",
        )

    def test_tiled_catalog_not_found(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("tiled catalog not found at cms/raw"),
            "Tiled access failure",
        )

    def test_tiled_auth_failure(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("tiled auth error: 401"),
            "Tiled access failure",
        )

    def test_output_permission_failure(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("permission denied: /nsls2/data/outputs"),
            "output permission failure",
        )

    def test_unsupported_config(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("unsupported tiling mode: triangle"),
            "unsupported config",
        )

    def test_unknown_failure(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure("something completely unexpected happened"),
            "unknown failure",
        )

    def test_empty_stderr(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure(""),
            "unknown failure",
        )

    def test_none_stderr(self):
        self.assertEqual(
            auto_stitch._categorize_anchor_failure(None),
            "unknown failure",
        )


if __name__ == "__main__":
    unittest.main()
