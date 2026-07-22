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
        return json.load(handle)
