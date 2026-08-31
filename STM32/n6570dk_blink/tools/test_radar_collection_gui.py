#!/usr/bin/env python3
"""Host-only checks for the GUI collector's CSV writer."""

import csv
from collections import deque
import importlib.util
import math
import struct
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from radar_collection_gui import (
    CLASS_NAMES,
    DEFAULT_TARGET_SELECTION_MODE,
    HOST_DEMO_MANIFEST,
    TARGET_SELECTION_AUTO,
    TARGET_SELECTION_FORCE_INDEX_1,
    TARGET_SELECTION_TEAMMATE,
    CausalHostInference,
    RadarCollectionApp,
    RadarTarget,
    SingleTarget20HzWriter,
    format_target_table_row,
    is_teammate_compatible_target,
    parse_teammate_target_chunk,
    rank_target,
    load_demo_binding,
    summarize_live_data_quality,
    teammate_next_sample_time,
)


def teammate_frame(targets: list[tuple[float, float, float, int, int]], *, checksums: bytes = b"\x00\x00") -> bytes:
    payload = struct.pack("<I", len(targets)) + b"".join(
        struct.pack("<fffii", x, y, z, dop_idx, cluster_id)
        for x, y, z, dop_idx, cluster_id in targets
    )
    header = b"\x01\x12\x34" + len(payload).to_bytes(2, "big") + b"\x0a\x04" + checksums[:1]
    return header + payload + checksums[1:2]


class SingleTargetWriterTests(unittest.TestCase):
    def test_historical_name_schema_cluster_and_rate(self) -> None:
        with TemporaryDirectory() as temporary:
            started = datetime(2026, 7, 14, 16, 4, 29)
            writer = SingleTarget20HzWriter(Path(temporary), "falling", started)
            writer.write(RadarTarget(1.0, 2.0, 3.0, -4), "falling")
            writer.write(RadarTarget(4.0, 5.0, 6.0, -7), "falling")
            writer.close()
            self.assertEqual(writer.path.name, "falling_20260714_160429_20hz.csv")
            with writer.path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(list(rows[0]), ["timestamp", "activity", "frame", "cluster_id", "x", "y", "z", "dop_idx"])
        self.assertEqual(rows[0]["cluster_id"], "1")
        self.assertEqual(rows[0]["frame"], "0")
        self.assertEqual(rows[1]["frame"], "1")
        self.assertEqual(rows[0]["timestamp"], "2026-07-14T16:04:29.000000")
        self.assertEqual(rows[1]["timestamp"], "2026-07-14T16:04:29.050000")

    def test_teammate_writer_is_lazy_and_uses_actual_sample_timestamps(self) -> None:
        with TemporaryDirectory() as temporary:
            started = datetime(2026, 7, 28, 17, 12, 55)
            writer = SingleTarget20HzWriter(
                Path(temporary), "sitting", started, teammate_compatible=True,
            )
            self.assertFalse(writer.path.exists())
            target = RadarTarget(0.4, 0.5, -0.4, 0)
            writer.write(
                target, "sitting", sampled_at=datetime(2026, 7, 28, 17, 12, 58, 69_742),
            )
            writer.write(
                target, "sitting", sampled_at=datetime(2026, 7, 28, 17, 12, 58, 120_457),
            )
            writer.close()
            with writer.path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["timestamp"], "2026-07-28T17:12:58.069742")
        self.assertEqual(rows[1]["timestamp"], "2026-07-28T17:12:58.120457")
        self.assertEqual(rows[0]["cluster_id"], "1")
        self.assertEqual(rows[1]["frame"], "1")
        self.assertEqual(rows[0]["x"], rows[1]["x"])

    def test_teammate_writer_leaves_no_empty_csv(self) -> None:
        with TemporaryDirectory() as temporary:
            writer = SingleTarget20HzWriter(
                Path(temporary), "standing", datetime(2026, 7, 28, 12, 0, 0),
                teammate_compatible=True,
            )
            writer.close()
            self.assertFalse(writer.path.exists())

    def test_teammate_parser_takes_first_target_and_ignores_protocol_checksums(self) -> None:
        first = (0.0, 2.0, 0.0, -3, 91)
        closer_second = (0.0, 0.5, 0.0, 7, 42)
        target = parse_teammate_target_chunk(
            b"noise" + teammate_frame([first, closer_second], checksums=b"\x00\x00")
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual((target.x, target.y, target.z, target.dop_idx), first[:4])
        self.assertEqual(target.target_num, 2)
        # The historical CSV path ignores the protocol cluster_id and writes 1.
        self.assertEqual(target.cluster_id, 1)

    def test_teammate_parser_matches_primary_reference_on_golden_chunks(self) -> None:
        reference_path = (
            Path(__file__).resolve().parents[3]
            / "DataCollection_Tools/collect_fixed_20hz.py"
        )
        if not reference_path.exists():
            self.skipTest("optional historical teammate collector is not in this repository")
        if "serial" not in sys.modules:
            sys.modules["serial"] = types.ModuleType("serial")
        spec = importlib.util.spec_from_file_location("teammate_reference_collector", reference_path)
        assert spec is not None and spec.loader is not None
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference = reference_module.FixedRate20HzCollector()

        accepted = teammate_frame([(0.0, 1.0, -0.5, -2, 77)])
        chunks = [
            accepted,
            b"prefix" + teammate_frame([
                (0.0, 2.0, 0.0, 4, 88),
                (0.0, 0.4, 0.0, 9, 99),
            ]),
            teammate_frame([(0.0, 8.1, 0.0, 0, 1)]),
            accepted[:12],
            teammate_frame([(0.0, 0.2, 0.0, 0, 1)]) + accepted,
        ]
        for chunk in chunks:
            with self.subTest(chunk_length=len(chunk)):
                old = reference.parse_target_data(chunk)
                new = parse_teammate_target_chunk(chunk)
                self.assertEqual(old is None, new is None)
                if old is not None and new is not None:
                    self.assertEqual(
                        (old["x"], old["y"], old["z"], old["dop_idx"], old["distance"]),
                        (new.x, new.y, new.z, new.dop_idx, new.distance),
                    )

    def test_teammate_parser_is_stateless_across_read_chunks(self) -> None:
        frame = teammate_frame([(0.0, 1.0, 0.0, 0, 5)])
        split = len(frame) // 2
        self.assertIsNone(parse_teammate_target_chunk(frame[:split]))
        self.assertIsNone(parse_teammate_target_chunk(frame[split:]))
        self.assertIsNotNone(parse_teammate_target_chunk(frame))

    def test_teammate_gates_match_legacy_boundaries_and_nan_quirk(self) -> None:
        self.assertTrue(is_teammate_compatible_target(RadarTarget(0.3, 0.0, 0.0, 0)))
        self.assertTrue(is_teammate_compatible_target(RadarTarget(8.0, 0.0, 0.0, 0)))
        self.assertTrue(is_teammate_compatible_target(RadarTarget(0.0, 1.0, -3.0, 0)))
        self.assertTrue(is_teammate_compatible_target(RadarTarget(0.0, 1.0, 2.0, 0)))
        self.assertFalse(is_teammate_compatible_target(RadarTarget(0.299, 0.0, 0.0, 0)))
        self.assertFalse(is_teammate_compatible_target(RadarTarget(8.001, 0.0, 0.0, 0)))
        self.assertTrue(is_teammate_compatible_target(RadarTarget(math.nan, 0.0, 0.0, 0)))

    def test_teammate_selection_never_substitutes_a_better_later_target(self) -> None:
        targets = [
            RadarTarget(0.0, 7.5, 0.0, 1, cluster_id=11),
            RadarTarget(0.0, 0.6, 0.0, 2, cluster_id=22),
        ]
        selection = rank_target(
            targets, mode=TARGET_SELECTION_TEAMMATE, previous_target=targets[1],
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual((selection.index, selection.target.cluster_id), (0, 11))

        # If index 0 fails the old gate, index 1 is not used as a fallback.
        self.assertIsNone(rank_target(
            [RadarTarget(0.0, 8.1, 0.0, 1), targets[1]],
            mode=TARGET_SELECTION_TEAMMATE,
            previous_target=None,
        ))

    def test_teammate_scheduler_drops_missed_ticks_instead_of_catching_up(self) -> None:
        self.assertAlmostEqual(teammate_next_sample_time(1.00, 1.00), 1.05)
        self.assertAlmostEqual(teammate_next_sample_time(1.00, 1.20), 1.25)

    def test_gui_compatibility_tick_holds_latest_target_without_tracking_timeout(self) -> None:
        class Variable:
            def __init__(self, value: str) -> None:
                self.value = value

            def get(self) -> str:
                return self.value

            def set(self, value: str) -> None:
                self.value = value

        with TemporaryDirectory() as temporary:
            app = object.__new__(RadarCollectionApp)
            app.collecting = True
            app.collection_starts_at = 0.0
            app.collection_deadline = 10.0
            app.next_sample_at = 1.0
            app.latest_target = RadarTarget(0.0, 1.0, 0.0, 0)
            app.activity_var = Variable("sitting")
            app.frames_var = Variable("0")
            app.sample_times = []
            app.writer = SingleTarget20HzWriter(
                Path(temporary), "sitting", datetime(2026, 7, 28, 12, 0, 0),
                teammate_compatible=True,
            )
            # Even when the normal GUI freshness result is false and the tick
            # is 200 ms late, compatibility mode writes one held row only.
            app._collect_due(now=1.2, tracking=False)
            self.assertEqual(app.writer.frame, 1)
            self.assertAlmostEqual(app.next_sample_at, 1.25)
            app.writer.close()

    def test_default_teammate_target_is_identical_for_display_csv_quality_and_inference(self) -> None:
        class Variable:
            def __init__(self, value: object = "") -> None:
                self.value = value

            def get(self) -> object:
                return self.value

            def set(self, value: object) -> None:
                self.value = value

        class FakePort:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            @property
            def in_waiting(self) -> int:
                return len(self.payload)

            def read(self, size: int) -> bytes:
                result, self.payload = self.payload[:size], self.payload[size:]
                return result

        self.assertEqual(DEFAULT_TARGET_SELECTION_MODE, TARGET_SELECTION_TEAMMATE)
        self.assertIsNotNone(CausalHostInference)
        self.assertIsNotNone(load_demo_binding)

        first = (0.25, 1.5, -0.5, -3, 91)
        closer_smart_candidate = (0.0, 0.6, 0.0, 7, 42)
        payload = teammate_frame([first, closer_smart_candidate])

        with TemporaryDirectory() as temporary:
            app = object.__new__(RadarCollectionApp)
            app.port = FakePort(payload)
            app.target_selection_mode_var = Variable(DEFAULT_TARGET_SELECTION_MODE)
            app.latest_target = None
            app.latest_frame_targetnum = None
            app.latest_frame_targets = []
            app.latest_selected_target_index = None
            app.last_selected_target_index = None
            app.last_selected_cluster_id = None
            app.previous_selected_target = None
            app.selection_reason = None
            app.selected_target_is_current = False
            app.selected_cluster_switches = 0
            app.selection_change_count = 0
            app.targetnum_dropouts = 0
            app.last_target_at = 0.0
            app.last_empty_target_list_at = 0.0
            app.next_teammate_inference_frame_id = 0
            app.quality_history = deque(maxlen=60)
            app.inference_queue = deque()
            app.last_inference_update = None
            app.next_inference_at = 10.0
            app.inference_startup_error = None
            app.inference = CausalHostInference(load_demo_binding(HOST_DEMO_MANIFEST))

            app.target_table_vars = [[Variable("—") for _ in range(8)] for _ in range(3)]
            app.selection_change_var = Variable("selection changes: 0")
            app.targetnum_var = Variable("—")
            app.cluster_id_var = Variable("—")
            app.x_var = Variable("—")
            app.y_var = Variable("—")
            app.z_var = Variable("—")
            app.distance_var = Variable("—")
            app.dop_var = Variable("—")
            app.raw_target_var = Variable("")
            app.used_target_var = Variable("")
            app.sample_freshness_var = Variable("")
            app.continuity_var = Variable("")
            app.quality_var = Variable("")
            app.quality_window_var = Variable("")
            app.inference_buffer_var = Variable("")
            app.inference_class_var = Variable("")
            app.inference_health_var = Variable("")
            app.inference_reset_var = Variable("")
            app.inference_reset_counts_var = Variable("")
            app.inference_missing_var = Variable("")
            app.inference_timing_var = Variable("")
            app.inference_mode_var = Variable("")
            app.probability_vars = {name: Variable("—") for name in CLASS_NAMES}

            app.collecting = True
            app.collection_starts_at = 10.0
            app.collection_deadline = 20.0
            app.next_sample_at = 10.0
            app.activity_var = Variable("walking")
            app.frames_var = Variable("0")
            app.sample_times = deque()
            app.writer = SingleTarget20HzWriter(
                Path(temporary), "walking", datetime(2026, 7, 30, 12, 0, 0),
                teammate_compatible=True,
            )

            with patch("radar_collection_gui.time.monotonic", return_value=10.0):
                app._decode_available()

            selected = app.latest_target
            self.assertIsNotNone(selected)
            assert selected is not None
            queued_target, queued_frame_id, queued_at = app.inference_queue[-1]
            self.assertIs(selected, app.quality_history[-1])
            self.assertIs(selected, queued_target)
            self.assertEqual(app.latest_selected_target_index, 0)
            self.assertEqual((selected.x, selected.y, selected.z, selected.dop_idx), first[:4])
            self.assertNotEqual((selected.x, selected.y, selected.z, selected.dop_idx), closer_smart_candidate[:4])
            self.assertEqual((queued_frame_id, queued_at), (0, 10.0))

            tracking = app._update_live_values(10.0)
            self.assertTrue(tracking)
            self.assertEqual(app.x_var.get(), f"{selected.x:.3f} m")
            self.assertIn("Teammate-compatible", str(app.used_target_var.get()))
            self.assertIn("index=0", str(app.used_target_var.get()))
            self.assertIn("distance", str(app.quality_var.get()))

            app._advance_inference(10.0, tracking)
            self.assertEqual(app.inference.buffer_progress, 1)
            inferred = app.inference._records[-1]
            self.assertEqual(
                (inferred.x, inferred.y, inferred.z, inferred.dop_idx),
                (selected.x, selected.y, selected.z, selected.dop_idx),
            )

            app._collect_due(10.0, tracking)
            app.writer.close()
            with app.writer.path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], "0")
        self.assertEqual(
            tuple(float(rows[0][name]) for name in ("x", "y", "z"))
            + (int(rows[0]["dop_idx"]),),
            (selected.x, selected.y, selected.z, selected.dop_idx),
        )

    def test_live_quality_ports_distance_and_z_limits_from_reference_validator(self) -> None:
        good = summarize_live_data_quality([RadarTarget(0.0, 1.0, 0.0, 0)])
        self.assertEqual(good.level, "GOOD")
        self.assertTrue(good.current_distance_in_range)
        self.assertTrue(good.current_z_in_range)

        distance_error = summarize_live_data_quality([RadarTarget(0.0, 6.1, 0.0, 0)])
        self.assertEqual(distance_error.level, "ERROR")
        self.assertFalse(distance_error.current_distance_in_range)

        z_error = summarize_live_data_quality([RadarTarget(0.0, 1.0, 1.1, 0)])
        self.assertEqual(z_error.level, "ERROR")
        self.assertFalse(z_error.current_z_in_range)

    def test_live_quality_reports_frequent_position_jumps(self) -> None:
        targets = [
            RadarTarget(0.0, 1.0, 0.0, 0),
            RadarTarget(0.0, 2.0, 0.0, 0),
            RadarTarget(0.0, 3.0, 0.0, 0),
        ]
        quality = summarize_live_data_quality(targets)
        self.assertEqual(quality.level, "ERROR")
        self.assertEqual(quality.jump_percent, 100.0)

    def test_auto_target_selection_prefers_continuity_then_nearest_valid(self) -> None:
        targets = [
            RadarTarget(0.0, 2.0, 0.0, 1, target_num=3, cluster_id=11),
            RadarTarget(0.0, 0.8, 0.0, 2, target_num=3, cluster_id=22),
            RadarTarget(0.0, 1.2, 0.0, 3, target_num=3, cluster_id=33),
        ]
        nearest = rank_target(targets, mode=TARGET_SELECTION_AUTO, previous_target=None)
        self.assertIsNotNone(nearest)
        assert nearest is not None
        self.assertEqual((nearest.index, nearest.reason), (1, "nearest valid"))

        continuity = rank_target(targets, mode=TARGET_SELECTION_AUTO, previous_target=targets[0])
        self.assertIsNotNone(continuity)
        assert continuity is not None
        self.assertEqual((continuity.index, continuity.reason), (0, "continuity"))

        forced = rank_target(targets, mode=TARGET_SELECTION_FORCE_INDEX_1, previous_target=targets[0])
        self.assertIsNotNone(forced)
        assert forced is not None
        self.assertEqual((forced.index, forced.reason), (1, "forced index 1 (debug)"))
        self.assertIsNone(rank_target([], mode=TARGET_SELECTION_AUTO, previous_target=None))

        tied = [RadarTarget(0.0, 1.0, 0.0, 0), RadarTarget(0.0, -1.0, 0.0, 0)]
        fallback = rank_target(tied, mode=TARGET_SELECTION_AUTO, previous_target=None)
        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertEqual((fallback.index, fallback.reason), (0, "fallback to first valid"))

        row = format_target_table_row(1, targets[1], used=True)
        self.assertEqual(row, ("USED", "1", "22", "0.000", "0.800", "0.000", "0.800", "2"))

    def test_auto_target_selection_excludes_strict_quality_failures(self) -> None:
        targets = [
            RadarTarget(0.0, 0.2, 0.0, 0, cluster_id=1),  # too close
            RadarTarget(0.0, 1.0, 1.2, 0, cluster_id=2),  # Z too high
            RadarTarget(0.0, 1.5, 0.0, 0, cluster_id=3),
        ]
        selection = rank_target(targets, mode=TARGET_SELECTION_AUTO, previous_target=None)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.index, 2)
        self.assertEqual(selection.reason, "nearest valid")


if __name__ == "__main__":
    unittest.main()
