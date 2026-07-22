import json
import os
from dataclasses import dataclass, field


@dataclass
class StitchSettings:
    enabled: bool
    verify_outputs: bool
    config: dict = field(default_factory=dict)


def load_stitch_settings() -> StitchSettings:
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
    """
    enabled = os.environ.get("ANCHOR_AUTOSTITCH_ENABLED", "false").lower() in ("1", "true", "yes")
    verify_outputs = os.environ.get("ANCHOR_AUTOSTITCH_VERIFY_OUTPUTS", "true").lower() not in ("0", "false", "no")
    config = json.loads(os.environ.get("ANCHOR_STITCH_CONFIG", "{}"))
    return StitchSettings(enabled=enabled, verify_outputs=verify_outputs, config=config)
