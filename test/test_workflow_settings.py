import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


import workflow_settings  # noqa: E402
from workflow_settings import StitchSettings, load_stitch_settings


class LoadStitchSettingsDefaultsTests(unittest.TestCase):
    def test_autostitch_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANCHOR_AUTOSTITCH_ENABLED", None)
            settings = load_stitch_settings()
        self.assertFalse(settings.enabled)

    def test_verify_outputs_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS", None)
            settings = load_stitch_settings()
        self.assertTrue(settings.verify_outputs)

    def test_empty_stitch_config_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANCHOR_STITCH_CONFIG", None)
            settings = load_stitch_settings()
        self.assertEqual(settings.config, {})


class LoadStitchSettingsEnabledTests(unittest.TestCase):
    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_ENABLED": "true"}):
            settings = load_stitch_settings()
        self.assertTrue(settings.enabled)

    def test_enabled_with_1(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_ENABLED": "1"}):
            settings = load_stitch_settings()
        self.assertTrue(settings.enabled)

    def test_enabled_with_yes(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_ENABLED": "yes"}):
            settings = load_stitch_settings()
        self.assertTrue(settings.enabled)

    def test_disabled_with_false(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_ENABLED": "false"}):
            settings = load_stitch_settings()
        self.assertFalse(settings.enabled)

    def test_disabled_with_0(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_ENABLED": "0"}):
            settings = load_stitch_settings()
        self.assertFalse(settings.enabled)


class LoadStitchSettingsVerifyTests(unittest.TestCase):
    def test_verify_disabled_with_false(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS": "false"}):
            settings = load_stitch_settings()
        self.assertFalse(settings.verify_outputs)

    def test_verify_disabled_with_0(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS": "0"}):
            settings = load_stitch_settings()
        self.assertFalse(settings.verify_outputs)

    def test_verify_disabled_with_no(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS": "no"}):
            settings = load_stitch_settings()
        self.assertFalse(settings.verify_outputs)

    def test_verify_enabled_with_true(self):
        with patch.dict(os.environ, {"ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS": "true"}):
            settings = load_stitch_settings()
        self.assertTrue(settings.verify_outputs)


class LoadStitchSettingsConfigTests(unittest.TestCase):
    def test_stitch_config_from_json(self):
        config_blob = json.dumps({"mode": "saxs", "threshold": 0.5})
        with patch.dict(os.environ, {"ANCHOR_STITCH_CONFIG": config_blob}):
            settings = load_stitch_settings()
        self.assertEqual(settings.config, {"mode": "saxs", "threshold": 0.5})

    def test_stitch_config_empty_json_object(self):
        with patch.dict(os.environ, {"ANCHOR_STITCH_CONFIG": "{}"}):
            settings = load_stitch_settings()
        self.assertEqual(settings.config, {})

    def test_stitch_config_rejects_malformed_json(self):
        with patch.dict(os.environ, {"ANCHOR_STITCH_CONFIG": "{"}):
            with self.assertRaisesRegex(ValueError, "valid JSON"):
                load_stitch_settings()

    def test_stitch_config_rejects_non_object_json(self):
        with patch.dict(os.environ, {"ANCHOR_STITCH_CONFIG": "[]"}):
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_stitch_settings()

    def test_workflow_options_override_environment(self):
        with patch.dict(
            os.environ,
            {
                "ANCHOR_AUTOSTITCH_ENABLED": "false",
                "ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS": "false",
                "ANCHOR_STITCH_CONFIG": "{}",
            },
        ):
            settings = load_stitch_settings(
                workflow_options={
                    "anchor_autostitch": {
                        "enabled": True,
                        "verify_outputs": True,
                        "config": {"max_lookback": 9},
                    }
                },
            )

        self.assertEqual(settings, StitchSettings(enabled=True, verify_outputs=True, config={"max_lookback": 9}))

    def test_rejects_non_mapping_anchor_options(self):
        with self.assertRaisesRegex(ValueError, "anchor_autostitch"):
            load_stitch_settings(workflow_options={"anchor_autostitch": True})

    def test_rejects_non_mapping_workflow_options(self):
        with self.assertRaisesRegex(ValueError, "workflow_options"):
            load_stitch_settings(workflow_options=True)


if __name__ == "__main__":
    unittest.main()
