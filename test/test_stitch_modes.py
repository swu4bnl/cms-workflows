import json
import tempfile
import unittest
from pathlib import Path

from stitch.core.config import load_config
from stitch.core import modes


class ConfigMaskPathTests(unittest.TestCase):
    def test_default_mask_paths_resolve_to_existing_files(self):
        config = load_config()
        detector_configs = config["detector"]["image_streams"]

        for detector_config in detector_configs:
            mask_path = Path(detector_config["mask_path"])
            self.assertTrue(mask_path.is_absolute())
            self.assertTrue(mask_path.is_file())


class ModeRegistryConfigPathTests(unittest.TestCase):
    def test_required_labels_are_loaded_per_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.json"
            second = tmp_path / "second.json"
            first.write_text(
                json.dumps({"modes": {"ygaps": {"required_labels": ["a", "b"]}}}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"modes": {"ygaps": {"required_labels": ["c", "d"]}}}),
                encoding="utf-8",
            )

            modes._mode_registry.cache_clear()

            self.assertEqual(modes.get_required_labels("ygaps", config_path=str(first)), ["a", "b"])
            self.assertEqual(modes.get_required_labels("ygaps", config_path=str(second)), ["c", "d"])


if __name__ == "__main__":
    unittest.main()