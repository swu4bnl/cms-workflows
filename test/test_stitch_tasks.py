import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so stitch_tasks.py can be imported without Prefect or Tiled
# ---------------------------------------------------------------------------

prefect_stub = types.ModuleType("prefect")


def task(func=None, *args, **kwargs):
    def decorator(func):
        return func

    if callable(func):
        return func
    return decorator


def get_run_logger():
    return MagicMock()


prefect_stub.task = task
prefect_stub.get_run_logger = get_run_logger
sys.modules.setdefault("prefect", prefect_stub)

data_validation_stub = types.ModuleType("data_validation")
data_validation_stub.get_run = lambda *args, **kwargs: None
sys.modules.setdefault("data_validation", data_validation_stub)


import stitch_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CategorizeAnchorFailureTests(unittest.TestCase):
    def test_missing_scan_range(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("missing scan range in args"),
            "missing scan range",
        )

    def test_start_scan_keyword(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("required arg: start_scan not provided"),
            "missing scan range",
        )

    def test_end_scan_keyword(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("end_scan must be specified"),
            "missing scan range",
        )

    def test_incomplete_groups(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("could not find all required tiles for mode=ygaps"),
            "incomplete groups",
        )

    def test_missing_required_tiles(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("missing required tiles: ['pos2']"),
            "incomplete groups",
        )

    def test_missing_detector_image_key(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("image_key 'pilatus2m-1_image' not found in primary stream"),
            "missing detector image key",
        )

    def test_tiled_catalog_not_found(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("tiled catalog not found at cms/raw"),
            "Tiled access failure",
        )

    def test_tiled_auth_failure(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("tiled auth error: 401"),
            "Tiled access failure",
        )

    def test_output_permission_failure(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("permission denied: /nsls2/data/outputs"),
            "output permission failure",
        )

    def test_unsupported_config(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("unsupported tiling mode: triangle"),
            "unsupported config",
        )

    def test_unknown_failure(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure("something completely unexpected happened"),
            "unknown failure",
        )

    def test_empty_stderr(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure(""),
            "unknown failure",
        )

    def test_none_stderr(self):
        self.assertEqual(
            stitch_tasks._categorize_anchor_failure(None),
            "unknown failure",
        )


class RunAutoStitchAnchorTests(unittest.TestCase):
    def test_calls_python_runner_directly(self):
        run = types.SimpleNamespace(
            start={
                "scan_id": "42",
                "cycle": "2026-1",
                "data_session": "pass-12345",
                "experiments_directory": "experiments",
                "experiment_alias_directory": "sample-a",
            }
        )
        runner = MagicMock(return_value={"output_dir": "/tmp/stitch-output"})
        stitch_module = types.ModuleType("stitch.runner")
        stitch_module.run_stitch_validation = runner

        with patch.dict(sys.modules, {"stitch.runner": stitch_module}), patch.object(
            stitch_tasks, "get_run", return_value=run
        ):
            result = stitch_tasks.run_auto_stitch_anchor(
                "uid-123",
                api_key="secret",
                stitch_config={
                    "max_lookback": 7,
                    "config_path": "configs/test.json",
                    "out_dir": "outputs/test",
                    "tiled_uri": "https://example.invalid",
                    "catalog_path": "cms/raw",
                },
            )

        runner.assert_called_once_with(
            anchor_scan=42,
            max_lookback=7,
            tiled_uri="https://example.invalid",
            catalog_path="cms/raw",
            config_path=str(stitch_tasks.STITCH_PACKAGE_DIR / "configs" / "test.json"),
            out_dir=str(stitch_tasks.STITCH_PACKAGE_DIR / "outputs" / "test"),
        )
        self.assertEqual(
            result,
            {"uid": "uid-123", "scan_id": 42, "output_dir": "/tmp/stitch-output", "plot": False},
        )

    def test_uses_experiment_alias_directory_as_default_output_dir(self):
        run = types.SimpleNamespace(
            start={
                "scan_id": "43",
                "cycle": "2026-1",
                "data_session": "pass-12345",
                "experiments_directory": "experiments",
                "experiment_alias_directory": "sample-a",
            }
        )
        runner = MagicMock(return_value={"output_dir": "/tmp/stitch-output"})
        stitch_module = types.ModuleType("stitch.runner")
        stitch_module.run_stitch_validation = runner

        with patch.dict(sys.modules, {"stitch.runner": stitch_module}), patch.object(
            stitch_tasks, "get_run", return_value=run
        ):
            stitch_tasks.run_auto_stitch_anchor("uid-456")

        runner.assert_called_once()
        self.assertEqual(
            runner.call_args.kwargs["out_dir"],
            "/nsls2/data/cms/proposals/2026-1/pass-12345/experiments/sample-a",
        )

    def test_returns_plot_flag_from_config(self):
        run = types.SimpleNamespace(
            start={
                "scan_id": "44",
                "cycle": "2026-1",
                "data_session": "pass-12345",
                "experiments_directory": "experiments",
                "experiment_alias_directory": "sample-a",
            }
        )
        runner = MagicMock(return_value={"output_dir": "/tmp/stitch-output"})
        stitch_module = types.ModuleType("stitch.runner")
        stitch_module.run_stitch_validation = runner

        with patch.dict(sys.modules, {"stitch.runner": stitch_module}), patch.object(
            stitch_tasks, "get_run", return_value=run
        ):
            result = stitch_tasks.run_auto_stitch_anchor("uid-789", stitch_config={"stitch_plot": True})

        self.assertTrue(result["plot"])


class VerifyStitchOutputsTests(unittest.TestCase):
    def test_verifies_required_outputs_without_preview_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "validation_index.json").write_text("{}", encoding="utf-8")
            (output_dir / "stitched.tiff").write_bytes(b"tiff")
            (output_dir / "stitched.json").write_text("{}", encoding="utf-8")

            result = stitch_tasks.verify_stitch_outputs({"output_dir": str(output_dir), "plot": False})

        self.assertEqual(result["tiff_count"], 1)
        self.assertEqual(result["sidecar_json_count"], 1)
        self.assertEqual(result["preview_png_count"], 0)

    def test_requires_preview_png_when_plot_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "validation_index.json").write_text("{}", encoding="utf-8")
            (output_dir / "stitched.tiff").write_bytes(b"tiff")
            (output_dir / "stitched.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                stitch_tasks.verify_stitch_outputs({"output_dir": str(output_dir), "plot": True})


if __name__ == "__main__":
    unittest.main()
