"""Compatibility wrapper for the stitch.py CLI.

Core stitching behavior lives in auto_stitch/stitch.py.
"""

import os
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stitch import main


if __name__ == "__main__":
    raise SystemExit(main())
