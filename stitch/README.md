# Stitch Package

This folder contains the minimum Part 2 stitching runtime copied into `cms-workflows` so the Prefect flow can run standalone from this repository.

Included components:

- `runner.py`
- `core/` package
- `configs/stitching_defaults.json`
- `configs/Dectris/` masks required by the default config

`stitch_tasks.py` imports `run_stitch_validation` from `stitch.runner` directly.
