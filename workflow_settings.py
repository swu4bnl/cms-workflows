import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class StitchSettings:
    enabled: bool
    verify_outputs: bool
    config: dict = field(default_factory=dict)


def _coerce_bool(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes")


def _load_config_blob(config_blob) -> dict:
    if config_blob in (None, ""):
        return {}
    if isinstance(config_blob, dict):
        return config_blob
    try:
        config = json.loads(config_blob)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("ANCHOR_STITCH_CONFIG must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("ANCHOR_STITCH_CONFIG must decode to a JSON object")
    return config


def load_stitch_settings(workflow_options=None) -> StitchSettings:
    """Load stitch behavior settings from environment variables.

    Environment variables
    ---------------------
    ANCHOR_AUTOSTITCH_ENABLED : str
        Enable anchor auto-stitching. Accepted truthy values: "1", "true", "yes".
        Defaults to disabled.
    ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS : str
        Verify stitch outputs after stitching. Accepted falsy values: "0", "false", "no".
        Defaults to enabled.
    ANCHOR_STITCH_CONFIG : str
        JSON blob with additional configuration for the stitching operation.
        Defaults to an empty dict.

    workflow_options : dict, optional
        Flow parameter overrides. Anchor auto-stitch options are read from the
        ``anchor_autostitch`` key to avoid growing the flow signature as more
        optional workflow integrations are added.
    """
    options = workflow_options or {}
    if not isinstance(options, Mapping):
        raise ValueError("workflow_options must be a mapping")
    stitch_options = options.get("anchor_autostitch", {})
    if stitch_options is None:
        stitch_options = {}
    if not isinstance(stitch_options, Mapping):
        raise ValueError("workflow_options.anchor_autostitch must be a mapping")

    enabled = _coerce_bool(
        stitch_options.get("enabled"),
        default=os.environ.get("ANCHOR_AUTOSTITCH_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
    verify_outputs = _coerce_bool(
        stitch_options.get("verify_outputs"),
        default=os.environ.get("ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS", "true").lower() not in ("0", "false", "no"),
    )
    config_blob = os.environ.get("ANCHOR_STITCH_CONFIG", "{}") if "config" not in stitch_options else stitch_options["config"]
    config = _load_config_blob(config_blob)
    return StitchSettings(enabled=enabled, verify_outputs=verify_outputs, config=config)
