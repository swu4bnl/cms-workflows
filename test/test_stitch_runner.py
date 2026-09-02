import sys
import types
import unittest
from unittest.mock import patch


tiled_stub = types.ModuleType("tiled")
tiled_client_stub = types.ModuleType("tiled.client")
tiled_queries_stub = types.ModuleType("tiled.queries")
tiled_client_stub.from_uri = lambda *args, **kwargs: None
tiled_queries_stub.Key = lambda *args, **kwargs: None
sys.modules.setdefault("tiled", tiled_stub)
sys.modules.setdefault("tiled.client", tiled_client_stub)
sys.modules.setdefault("tiled.queries", tiled_queries_stub)


from stitch import runner  # noqa: E402


def _make_runs(scan_id_labels):
    """Build fake Tiled runs {scan_id: run} sharing group-a/ygaps, one label each."""
    return {
        scan_id: types.SimpleNamespace(
            start={
                "scan_id": scan_id,
                "uid": f"uid-{scan_id}",
                "stitch_group_id": "group-a",
                "stitch_tiling_mode": "ygaps",
                "stitch_tile_label": label,
            }
        )
        for scan_id, label in scan_id_labels
    }


def _fetch_anchor_group_via_lookback(runs, **kwargs):
    """Call _fetch_anchor_group with stitch_group_id search disabled, forcing the scan_id lookback path."""

    def search_by_key(node, key, value):
        if key == "scan_id":
            return [runs[value]] if value in runs else []
        return []  # stitch_group_id search finds nothing, forcing the lookback fallback

    with patch.object(runner, "_open_catalog", return_value=object()), patch.object(
        runner, "_search_by_key", side_effect=search_by_key
    ), patch.object(runner, "get_required_labels", return_value=["pos1", "pos2"]):
        return runner._fetch_anchor_group(
            tiled_uri="https://example.invalid",
            catalog_path="cms/raw",
            config_path="config.json",
            **kwargs,
        )


class AnchorRunSelectionTests(unittest.TestCase):
    def test_anchor_mode_selects_matching_repetition_from_block_ordered_tiles(self):
        runs = _make_runs(
            [
                (2440898, "pos1"),
                (2440899, "pos1"),
                (2440900, "pos1"),
                (2440901, "pos1"),
                (2440902, "pos2"),
                (2440903, "pos2"),
                (2440904, "pos2"),
                (2440905, "pos2"),
            ]
        )
        selected_runs, scan_range = _fetch_anchor_group_via_lookback(
            runs, anchor_scan=2440905, anchor_uid=None, max_lookback=8
        )

        self.assertEqual([run.start["scan_id"] for run in selected_runs], [2440901, 2440905])
        self.assertEqual(scan_range, [2440901, 2440905])

    def test_anchor_mode_can_stitch_incomplete_final_block_when_anchor_repetition_is_complete(self):
        runs = _make_runs(
            [
                (2440898, "pos1"),
                (2440899, "pos1"),
                (2440900, "pos1"),
                (2440901, "pos1"),
                (2440902, "pos2"),
                (2440903, "pos2"),
                (2440904, "pos2"),
            ]
        )
        selected_runs, scan_range = _fetch_anchor_group_via_lookback(
            runs, anchor_scan=2440904, anchor_uid=None, max_lookback=7
        )

        self.assertEqual([run.start["scan_id"] for run in selected_runs], [2440900, 2440904])
        self.assertEqual(scan_range, [2440900, 2440904])

    def test_anchor_mode_waits_when_anchor_is_not_final_tile_label(self):
        runs = _make_runs([(2440898, "pos1"), (2440899, "pos1"), (2440900, "pos1")])
        with self.assertRaisesRegex(RuntimeError, "not the final required tile"):
            _fetch_anchor_group_via_lookback(runs, anchor_scan=2440900, anchor_uid=None, max_lookback=3)

    def test_anchor_scan_rejects_duplicate_scan_id_matches(self):
        duplicate_runs = [
            types.SimpleNamespace(start={"scan_id": 42, "uid": "uid-a"}),
            types.SimpleNamespace(start={"scan_id": 42, "uid": "uid-b"}),
        ]
        with patch.object(runner, "_open_catalog", return_value=object()), patch.object(
            runner, "_search_by_key", return_value=duplicate_runs
        ):
            with self.assertRaisesRegex(RuntimeError, "pass anchor_uid instead"):
                runner._fetch_anchor_group(
                    tiled_uri="https://example.invalid",
                    catalog_path="cms/raw",
                    config_path="config.json",
                    anchor_scan=42,
                    anchor_uid=None,
                    max_lookback=3,
                )


if __name__ == "__main__":
    unittest.main()