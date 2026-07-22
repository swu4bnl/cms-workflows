"""Core typed models used by the stitching package."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TileMetadata:
    """Normalized metadata for one tile image."""

    stitch_group_id: str
    stitch_tiling_mode: str
    stitch_tile_label: str
    stitch_tile_index: int
    stitch_tile_total: Optional[int] = None
    detector_readback: Dict[str, float] = field(default_factory=dict)
    mask_path: Optional[str] = None
    source_uid: Optional[str] = None
    source_scan_id: Optional[int] = None
    source_filename: Optional[str] = None


@dataclass
class TileEntry:
    """Image array and normalized metadata for one tile."""

    image: np.ndarray
    metadata: TileMetadata
    mask: Optional[np.ndarray] = None


@dataclass
class StitchResult:
    """Output from the stitcher core."""

    stitched_image: np.ndarray
    result_metadata: Dict[str, Any]
