import json
import tempfile
import unittest
from pathlib import Path

from stitch.core import modes


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