import argparse
import json
import os
import subprocess
import sys
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tiled.client import from_uri
from tiled.queries import Key

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "stitching_defaults.json")
FALLBACK_OUTPUT_DIR = "StitchingOutputs"
FALLBACK_INDEX_PATH = os.path.join(FALLBACK_OUTPUT_DIR, "validation_index.json")


def load_settings(config_path=None):
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def config_value(settings, section, key, default=None):
    value = settings.get(section, {}).get(key, default)
    return default if value is None else value


def output_dir_from_config(settings):
    return config_value(settings, "outputs", "directory", FALLBACK_OUTPUT_DIR)


def index_path_from_config(settings):
    output_dir = output_dir_from_config(settings)
    index_filename = config_value(settings, "outputs", "index_filename", "validation_index.json")
    return os.path.join(output_dir, index_filename)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def safe_name(text):
    bad = '<>:"/\\|?* '
    out = str(text)
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def stitch_scans(
    start_scan,
    end_scan,
    out_dir,
    tiled_uri,
    catalog_path,
    config_path,
    anchor_scan=None,
    anchor_uid=None,
    max_lookback=50,
):
    print(f"\n{'='*60}")
    print(f"Stitching scans {start_scan}-{end_scan}...")
    print(f"{'='*60}")
    
    cmd = [
        "pixi", "run", "python",
        "examples/phase1_scan_range_from_tiled.py",
        "--tiled-uri", tiled_uri,
        "--catalog-path", catalog_path,
        "--out-dir", out_dir,
        "--config", config_path,
    ]

    if anchor_scan is not None or anchor_uid is not None:
        if anchor_scan is not None:
            cmd.extend(["--anchor-scan", str(anchor_scan)])
        if anchor_uid is not None:
            cmd.extend(["--anchor-uid", str(anchor_uid)])
        cmd.extend(["--max-lookback", str(int(max_lookback))])
    else:
        cmd.extend([
            "--start-scan", str(start_scan),
            "--end-scan", str(end_scan),
        ])
    
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=False)
    return result.returncode == 0


def load_index(index_path):
    path = resolve_path(index_path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def list_groups(index_path):
    index_payload, path = load_index(index_path)
    groups = index_payload.get("groups", [])
    print(f"Index: {path}")
    print(f"Scan range: {index_payload.get('scan_range')}")
    print(f"Groups: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"  {i}: {group.get('group_id')}")
        print(f"     image: {group.get('image') or group.get('tiff') or group.get('npz')}")
    return groups


def extract_start_doc(run):
    if hasattr(run, "start"):
        try:
            return dict(run.start)
        except Exception:
            pass
    metadata = getattr(run, "metadata", {})
    if isinstance(metadata, Mapping) and "start" in metadata:
        return dict(metadata["start"])
    return {}


def detector_config_for_group(group, detector_configs):
    detector_name = group.get("detector")
    if detector_name:
        for detector_config in detector_configs:
            if detector_config.get("name") == detector_name:
                return detector_config

    group_id = str(group.get("group_id", ""))
    for detector_config in detector_configs:
        name = str(detector_config.get("name", ""))
        if name and f"::{name}::" in group_id:
            return detector_config
    return {}


def extract_image(run, detector_config=None):
    if hasattr(run, "primary"):
        primary = run.primary.read()
    else:
        primary = run["primary"]["data"].read()

    candidates = [key for key in primary.data_vars if str(key).endswith("_image")]
    if not candidates:
        candidates = list(primary.data_vars)

    if detector_config:
        image_key = str(detector_config.get("image_key"))
        candidates = [key for key in candidates if str(key) == image_key]
        if not candidates:
            raise KeyError(f"Configured image_key {image_key!r} was not found in primary stream")

    image = np.asarray(primary[candidates[0]])
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[0]
    return np.asarray(image, dtype=float)


def select_stitch_run(result):
    matches = [result[key] for key in result]
    return next(
        (candidate for candidate in matches if extract_start_doc(candidate).get("stitch_group_id")),
        matches[0],
    )


def to_log_display(image):
    return np.log10(np.clip(np.abs(np.asarray(image, dtype=float)), 1e-3, None))


def build_signal_mask(image):
    image = np.asarray(image)
    finite = np.isfinite(image)
    if not np.any(finite):
        return finite

    background_value = float(np.min(image[finite]))
    if background_value < 0:
        return finite & (image > background_value + 1e-12)
    return finite & (image != 0)


def compute_signal_bbox(image, padding=40):
    rows, cols = np.nonzero(image)
    if rows.size == 0 or cols.size == 0:
        return None

    row_min = max(int(rows.min()) - padding, 0)
    row_max = min(int(rows.max()) + padding + 1, image.shape[0])
    col_min = max(int(cols.min()) - padding, 0)
    col_max = min(int(cols.max()) + padding + 1, image.shape[1])
    return row_min, row_max, col_min, col_max


def load_stitched_image(group):
    if group.get("image") and str(group.get("image_format", "")).lower() in {"tif", "tiff"}:
        return np.asarray(Image.open(resolve_path(group["image"])), dtype=float)
    if group.get("image") and str(group.get("image_format", "")).lower() == "npz":
        return np.load(resolve_path(group["image"]))["stitched_image"]
    if group.get("tiff"):
        return np.asarray(Image.open(resolve_path(group["tiff"])), dtype=float)
    if group.get("npz"):
        return np.load(resolve_path(group["npz"]))["stitched_image"]
    raise KeyError("Group does not contain a stitched image path ('tiff' or legacy 'npz').")


def sample_name_from_group_id(group_id):
    parts = str(group_id).split("::")
    if len(parts) >= 3:
        return "::".join(parts[2:])
    return str(group_id)


def collect_source_tiles(index_payload, group):
    target_group = group["group_id"]
    detector_configs = index_payload.get("settings", {}).get("detector_image_streams", [])
    detector_config = detector_config_for_group(group, detector_configs)
    detector_name = detector_config.get("name") or group.get("detector")
    client = from_uri(index_payload["tiled_uri"])
    node = client
    for part in index_payload["catalog_path"].split("/"):
        node = node[part]

    scan_start, scan_end = index_payload["scan_range"]
    tiles = []
    for scan_id in range(scan_start, scan_end + 1):
        result = node.search(Key("scan_id") == int(scan_id))
        if len(result) < 1:
            continue

        run = select_stitch_run(result)
        start = extract_start_doc(run)
        if not start.get("stitch_group_id"):
            continue

        group_key = f"{start.get('stitch_group_id')}::{detector_name}::{start.get('sample_name') or 'unknown_sample'}"
        if not (group_key == target_group or target_group.startswith(group_key + "__rep")):
            continue

        tile_index = int(start.get("stitch_tile_index", 0))
        tile_label = start.get("stitch_tile_label", f"tile{tile_index}")
        source_scan_id = start.get("scan_id", scan_id)
        tiles.append((tile_index, tile_label, source_scan_id, extract_image(run, detector_config)))

    return sorted(tiles, key=lambda item: item[0])


def plot_group(group_index=0, index_path=FALLBACK_INDEX_PATH, save_path=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if group_index < 0 or group_index >= len(groups):
        raise ValueError(f"group index must be between 0 and {len(groups) - 1}")

    group = groups[group_index]
    target_group = group["group_id"]
    tiles = collect_source_tiles(index_payload, group)
    stitched = load_stitched_image(group)
    tile_summary = ", ".join([f"{label}[{tile_index}]" for tile_index, label, _, _ in tiles]) or "none"

    tile_displays = [to_log_display(image) for _, _, _, image in tiles]
    display_img = to_log_display(stitched)
    finite_chunks = [image[np.isfinite(image)] for image in tile_displays + [display_img]]
    finite_chunks = [chunk for chunk in finite_chunks if chunk.size > 0]
    if finite_chunks:
        global_signal = np.concatenate(finite_chunks)
        shared_vmin, shared_vmax = np.percentile(global_signal, global_percentiles)
        if shared_vmin == shared_vmax:
            shared_vmin = float(global_signal.min())
            shared_vmax = float(global_signal.max())
    else:
        shared_vmin = -3.0
        shared_vmax = 1.0

    signal_mask = build_signal_mask(stitched)
    masked_display = np.ma.masked_where(~signal_mask, display_img)
    bbox = compute_signal_bbox(signal_mask.astype(np.uint8))

    n_panels = len(tiles) + 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = list(np.atleast_1d(axes))

    for i, (tile_index, label, source_scan_id, image) in enumerate(tiles):
        axes[i].imshow(tile_displays[i], cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
        axes[i].set_title(f"Before: {label}\nscan {source_scan_id} (index {tile_index})")
        axes[i].axis("off")

    image_after = axes[-2].imshow(display_img, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    axes[-2].set_title(f"After: stitched full canvas\nlabels: {tile_summary}")
    axes[-2].axis("off")

    axes[-1].imshow(masked_display, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    if bbox is not None:
        row_min, row_max, col_min, col_max = bbox
        axes[-1].set_xlim(col_min, col_max)
        axes[-1].set_ylim(row_max, row_min)
    axes[-1].set_title(f"After: cropped signal\nlabels: {tile_summary}")
    axes[-1].axis("off")

    if show_colorbar:
        colorbar = fig.colorbar(image_after, ax=axes, shrink=0.85, pad=0.01)
        colorbar.set_label("log10(|intensity|)")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(index_abs_path), f"preview_group{group_index}_{safe_name(target_group)}.png")
    save_abs_path = resolve_path(save_path)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_abs_path, dpi=150)
    plt.close(fig)

    print(f"Group: {target_group}")
    print(f"Tiles shown: {len(tiles)}")
    print(f"Stitched shape: {stitched.shape}")
    print(f"Saved preview: {save_abs_path}")
    return save_abs_path


def plot_overview(index_path=FALLBACK_INDEX_PATH, out_dir=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if out_dir is None:
        out_dir = os.path.dirname(index_abs_path)

    if not groups:
        raise ValueError("No stitched groups found in index.")

    detectors = []
    samples = []
    images = {}
    display_values = []
    for group in groups:
        detector = str(group.get("detector") or "unknown_detector")
        sample = sample_name_from_group_id(group.get("group_id", "unknown_sample"))
        if detector not in detectors:
            detectors.append(detector)
        if sample not in samples:
            samples.append(sample)

        stitched = load_stitched_image(group)
        display = to_log_display(stitched)
        images[(sample, detector)] = display
        finite = display[np.isfinite(display)]
        if finite.size:
            display_values.append(finite)

    if display_values:
        shared_vmin, shared_vmax = np.percentile(np.concatenate(display_values), global_percentiles)
    else:
        shared_vmin, shared_vmax = -3.0, 1.0
    if shared_vmin == shared_vmax:
        shared_vmin, shared_vmax = None, None

    fig, axes = plt.subplots(
        len(samples),
        len(detectors),
        figsize=(5 * len(detectors), 4.5 * len(samples)),
        squeeze=False,
    )
    image_after = None
    for row, sample in enumerate(samples):
        for col, detector in enumerate(detectors):
            ax = axes[row][col]
            display = images.get((sample, detector))
            if display is None:
                ax.axis("off")
                ax.set_title(f"{detector}\n{sample}\nmissing")
                continue
            image_after = ax.imshow(display, cmap="magma", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
            ax.set_title(f"{detector}\n{sample}")
            ax.axis("off")

    if show_colorbar and image_after is not None:
        colorbar = fig.colorbar(image_after, ax=axes, shrink=0.85, pad=0.01)
        colorbar.set_label("log10(|intensity|)")

    save_path = os.path.join(out_dir, "preview_all_stitched_outputs.png")
    save_abs_path = resolve_path(save_path)
    os.makedirs(os.path.dirname(save_abs_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_abs_path, dpi=150)
    plt.close(fig)
    print(f"Saved overview preview: {save_abs_path}")
    return save_abs_path


def plot_all(index_path=FALLBACK_INDEX_PATH, out_dir=None, global_percentiles=(1.0, 99.5), show_colorbar=False):
    index_payload, index_abs_path = load_index(index_path)
    groups = index_payload.get("groups", [])
    if out_dir is None:
        out_dir = os.path.dirname(index_abs_path)

    saved = []
    saved.append(plot_overview(index_path=index_path, out_dir=out_dir, global_percentiles=global_percentiles, show_colorbar=show_colorbar))
    for group_index, group in enumerate(groups):
        save_path = os.path.join(out_dir, f"preview_group{group_index}_{safe_name(group['group_id'])}.png")
        saved.append(
            plot_group(
                group_index=group_index,
                index_path=index_path,
                save_path=save_path,
                global_percentiles=global_percentiles,
                show_colorbar=show_colorbar,
            )
        )
    print(f"Saved {len(saved)} preview images.")
    return saved


def choose(value, default):
    return default if value is None else value


def run_pipeline(args):
    settings = load_settings(args.config)
    out_dir = choose(args.out_dir, output_dir_from_config(settings))
    index_path = os.path.join(out_dir, config_value(settings, "outputs", "index_filename", "validation_index.json"))
    tiled = settings.get("tiled", {})
    plotting = settings.get("plotting", {})
    global_percentiles = plotting.get("global_percentiles", [1.0, 99.5])

    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))

    os.makedirs(resolve_path(out_dir), exist_ok=True)
    if not stitch_scans(
        args.start_scan,
        args.end_scan,
        out_dir=out_dir,
        tiled_uri=choose(args.tiled_uri, tiled.get("uri", "https://tiled.nsls2.bnl.gov")),
        catalog_path=choose(args.catalog_path, tiled.get("catalog_path", "cms/raw")),
        config_path=args.config,
    ):
        print("Stitching failed.")
        return 1

    list_groups(index_path)
    if args.plot == "all":
        plot_all(index_path, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    elif args.plot is not None:
        plot_group(
            group_index=int(args.plot),
            index_path=index_path,
            global_percentiles=global_percentiles,
            show_colorbar=show_colorbar,
        )
    return 0


def run_anchor_pipeline(args):
    settings = load_settings(args.config)
    out_dir = choose(args.out_dir, output_dir_from_config(settings))
    index_path = os.path.join(out_dir, config_value(settings, "outputs", "index_filename", "validation_index.json"))
    tiled = settings.get("tiled", {})
    plotting = settings.get("plotting", {})
    global_percentiles = plotting.get("global_percentiles", [1.0, 99.5])

    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))

    os.makedirs(resolve_path(out_dir), exist_ok=True)
    if not stitch_scans(
        None,
        None,
        out_dir=out_dir,
        tiled_uri=choose(args.tiled_uri, tiled.get("uri", "https://tiled.nsls2.bnl.gov")),
        catalog_path=choose(args.catalog_path, tiled.get("catalog_path", "cms/raw")),
        config_path=args.config,
        anchor_scan=args.scan_id,
        anchor_uid=args.uid,
        max_lookback=args.max_lookback,
    ):
        print("Stitching failed.")
        return 1

    list_groups(index_path)
    if args.plot == "all":
        plot_all(index_path, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    elif args.plot is not None:
        plot_group(
            group_index=int(args.plot),
            index_path=index_path,
            global_percentiles=global_percentiles,
            show_colorbar=show_colorbar,
        )
    return 0


def list_command(args):
    settings = load_settings(args.config)
    index_path = choose(args.index_path, index_path_from_config(settings))
    list_groups(index_path)
    return 0


def plot_command(args):
    settings = load_settings(args.config)
    plotting = settings.get("plotting", {})
    index_path = choose(args.index_path, index_path_from_config(settings))
    global_percentiles = choose(args.global_percentiles, plotting.get("global_percentiles", [1.0, 99.5]))
    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))
    plot_group(
        group_index=args.group_index,
        index_path=index_path,
        save_path=args.save,
        global_percentiles=global_percentiles,
        show_colorbar=show_colorbar,
    )
    return 0


def plot_all_command(args):
    settings = load_settings(args.config)
    plotting = settings.get("plotting", {})
    index_path = choose(args.index_path, index_path_from_config(settings))
    global_percentiles = choose(args.global_percentiles, plotting.get("global_percentiles", [1.0, 99.5]))
    show_colorbar = choose(args.show_colorbar, bool(plotting.get("show_colorbar", False)))
    plot_all(index_path, out_dir=args.out_dir, global_percentiles=global_percentiles, show_colorbar=show_colorbar)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Single entry point for stitching and plotting Tiled scan ranges.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="stitch a scan range")
    run_parser.add_argument("start_scan", type=int)
    run_parser.add_argument("end_scan", type=int)
    run_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument(
        "--plot",
        nargs="?",
        const="all",
        default=None,
        help="plot all groups; optionally pass one group index, or use 'all' explicitly",
    )
    run_parser.add_argument("--show-colorbar", action="store_true", default=None)
    run_parser.add_argument("--out-dir", default=None)
    run_parser.add_argument("--tiled-uri", default=None)
    run_parser.add_argument("--catalog-path", default=None)
    run_parser.set_defaults(func=run_pipeline)

    run_anchor_parser = subparsers.add_parser(
        "run-anchor",
        help="stitch one acquisition by searching backward from a scan_id or uid",
    )
    anchor_group = run_anchor_parser.add_mutually_exclusive_group(required=True)
    anchor_group.add_argument("--scan-id", type=int, default=None)
    anchor_group.add_argument("--uid", default=None)
    run_anchor_parser.add_argument("--max-lookback", type=int, default=50)
    run_anchor_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run_anchor_parser.add_argument(
        "--plot",
        nargs="?",
        const="all",
        default=None,
        help="plot all groups; optionally pass one group index, or use 'all' explicitly",
    )
    run_anchor_parser.add_argument("--show-colorbar", action="store_true", default=None)
    run_anchor_parser.add_argument("--out-dir", default=None)
    run_anchor_parser.add_argument("--tiled-uri", default=None)
    run_anchor_parser.add_argument("--catalog-path", default=None)
    run_anchor_parser.set_defaults(func=run_anchor_pipeline)

    list_parser = subparsers.add_parser("list", help="list groups in a validation index")
    list_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    list_parser.add_argument("--index-path", default=None)
    list_parser.set_defaults(func=list_command)

    plot_parser = subparsers.add_parser("plot", help="plot one stitched output group")
    plot_parser.add_argument("group_index", type=int, nargs="?", default=0)
    plot_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    plot_parser.add_argument("--index-path", default=None)
    plot_parser.add_argument("--save", default=None)
    plot_parser.add_argument("--show-colorbar", action="store_true", default=None)
    plot_parser.add_argument("--global-percentiles", nargs=2, type=float, default=None)
    plot_parser.set_defaults(func=plot_command)

    plot_all_parser = subparsers.add_parser("plot-all", help="plot every group in a validation index")
    plot_all_parser.add_argument("--cionfig", default=DEFAULT_CONFIG_PATH)
    plot_all_parser.add_argument("--index-path", default=None)
    plot_all_parser.add_argument("--out-dir", default=None)
    plot_all_parser.add_argument("--show-colorbar", action="store_true", default=None)
    plot_all_parser.add_argument("--global-percentiles", nargs=2, type=float, default=None)
    plot_all_parser.set_defaults(func=plot_all_command)

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0].isdigit() and argv[1].isdigit():
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
