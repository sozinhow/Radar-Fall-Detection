#!/usr/bin/env python3
"""Host-only tests for decoded_csv_to_edge_impulse.py."""

from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoded_csv_to_edge_impulse import convert_rows


def target(offset: str, frame_id: str, cluster_id: str, x: str = "1.0") -> dict[str, str]:
    return {
        "stream_offset": offset,
        "frame_id": frame_id,
        "message_type": "0x0A04",
        "report": "target_list",
        "cluster_id": cluster_id,
        "x_m": x,
        "y_m": "2.0",
        "z_m": "3.0",
        "dop_idx": "-1",
    }


class ConversionTests(unittest.TestCase):
    def test_first_target_per_frame_has_fixed_cluster_and_20hz_timestamp(self) -> None:
        rows = [
            target("0", "1", "3"),
            target("0", "1", "4", "1.5"),
            {**target("20", "2", "9"), "report": "point_cloud"},
            target("40", "3", "7"),
        ]
        converted, skipped = convert_rows(rows, "falling", datetime.fromisoformat("2026-07-28T13:45:00"))
        self.assertEqual(skipped, 0)
        self.assertEqual([item["frame"] for item in converted], [0, 1])
        self.assertEqual(converted[0]["timestamp"], "2026-07-28T13:45:00.000000")
        self.assertEqual(converted[1]["timestamp"], "2026-07-28T13:45:00.050000")
        self.assertEqual(converted[0]["cluster_id"], 1)
        self.assertEqual(converted[0]["x"], "1.0")

    def test_empty_target_report_is_skipped_without_a_frame_gap(self) -> None:
        empty = target("0", "1", "")
        empty.update({"x_m": "", "y_m": "", "z_m": "", "dop_idx": ""})
        converted, skipped = convert_rows([empty, target("10", "2", "5")], "standing", datetime(2026, 7, 28))
        self.assertEqual(skipped, 1)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["frame"], 0)


if __name__ == "__main__":
    unittest.main()
