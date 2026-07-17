# Bundled Auto-Stitch Module

This folder contains the minimum Part 2 stitching runtime copied into `cms-workflows` so the Prefect flow can run standalone from this repository.

Included components:

- `stitch.py`
- `examples/phase1_scan_range_from_tiled.py`
- `stitching/` package
- `configs/stitching_defaults.json`
- `configs/Dectris/` masks required by the default config

`end_of_run_workflow.py` uses this bundled folder by default when `stitch_repo_path` is not set.
