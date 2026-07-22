"""Mode registry and validation rules for tiling modes."""

from functools import lru_cache
from typing import Dict, List

from .config import load_config
from .errors import MetadataMismatchError, MissingTileError


@lru_cache(maxsize=None)
def _mode_registry(config_path: str | None = None) -> Dict[str, Dict[str, List[str]]]:
    config = load_config(config_path)
    modes = config.get("modes", {})
    tiling_configs = config.get("tiling_configs", {})
    if isinstance(tiling_configs, dict) and tiling_configs:
        modes = {
            mode: {"required_labels": [entry.get("label") for entry in entries]}
            for mode, entries in tiling_configs.items()
        }
    if not isinstance(modes, dict) or not modes:
        raise MetadataMismatchError("Configuration is missing stitching mode definitions")
    return modes


def get_required_labels(mode: str, config_path: str | None = None) -> List[str]:
    registry = _mode_registry(config_path)
    if mode not in registry:
        raise MissingTileError(f"Unsupported tiling mode: {mode}")

    required = registry[mode].get("required_labels")
    if not isinstance(required, list) or not required:
        raise MetadataMismatchError(
            f"Mode {mode} configuration is missing required_labels"
        )
    return [str(label) for label in required]


def validate_mode_labels(mode: str, labels: List[str], config_path: str | None = None) -> None:
    required = get_required_labels(mode, config_path=config_path)
    missing = [label for label in required if label not in labels]
    if missing:
        raise MissingTileError(
            f"Mode {mode} is missing required tiles: {missing}. Got labels: {labels}"
        )
