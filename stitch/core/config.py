"""External configuration loading for the stitching pipeline."""

import json
import os
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "stitching_defaults.json")
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    package_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    for detector_config in config.get("detector", {}).get("image_streams", []):
        mask_path = detector_config.get("mask_path")
        if mask_path and not os.path.isabs(mask_path):
            detector_config["mask_path"] = os.path.normpath(os.path.join(package_root, mask_path))

    return config
