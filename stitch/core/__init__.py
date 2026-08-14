"""Modular auto-stitching package for tiled X-ray images."""

from .core import stitch_tiles
from .grouping import group_tile_entries
from .serialize import result_to_serializable
from .models import TileEntry, TileMetadata, StitchResult

__all__ = [
    "stitch_tiles",
    "group_tile_entries",
    "result_to_serializable",
    "TileEntry",
    "TileMetadata",
    "StitchResult",
]
