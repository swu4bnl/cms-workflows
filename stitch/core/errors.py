"""Custom exceptions for the stitching pipeline."""


class StitchingError(Exception):
    """Base error for stitching pipeline failures."""


class MissingTileError(StitchingError):
    """Raised when one or more expected tiles are missing from a group."""


class MetadataMismatchError(StitchingError):
    """Raised when metadata are inconsistent within a stitch group."""


class MissingMaskError(StitchingError):
    """Raised when a required mask is not provided."""


class DataLoadError(StitchingError):
    """Raised when image data cannot be loaded."""
