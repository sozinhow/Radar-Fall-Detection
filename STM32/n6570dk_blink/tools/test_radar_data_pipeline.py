"""Host-only tests for the receive-only radar data pipeline."""

from __future__ import annotations

import csv
import io
import struct
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hlk_ld6002b_capture import OutputWriter, tf_checksum  # noqa: E402
from radar_data_pipeline import (  # noqa: E402
    RCAP_TYPE_RAW,
    RcapParser,
    StreamDecoder,
    _checksum,
    _write_training_csv,
)


def tinyframe(frame_id: int, message_type: int, payload: bytes) -> bytes:
    header = (
        b"\x01"
        + frame_id.to_bytes(2, "big")
        + len(payload).to_bytes(2, "big")
        + message_type.to_bytes(2, "big")
    )
    return header + bytes([tf_checksum(header)]) + payload + bytes([tf_checksum(payload)])


def target_frame(frame_id: int = 1, x: float = 1.0) -> bytes:
    payload = struct.pack("<i", 1) + struct.pack("<fffii", x, 2.0, 3.0, -4, 9)
    return tinyframe(frame_id, 0x0A04, payload)


def rcap_record(payload: bytes, record_type: int = RCAP_TYPE_RAW) -> bytes:
    header = (
        b"\xA5\x5A"
        + bytes([1, record_type])
        + len(payload).to_bytes(2, "little")
        + (7).to_bytes(4, "little")
        + (1234).to_bytes(4, "little")
    )
    return header + payload + bytes([_checksum(header + payload)])


class RcapParserTests(unittest.TestCase):
    def test_split_record_and_invalid_record_recover(self) -> None:
        parser = RcapParser()
        encoded = rcap_record(b"hello")
        self.assertEqual(parser.feed(encoded[:1]), [])
        self.assertEqual(parser.feed(encoded[1:6]), [])
        records = parser.feed(encoded[6:])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].payload, b"hello")

    def test_valid_record_payload_is_unwrapped(self) -> None:
        parser = RcapParser()
        records = parser.feed(b"noise" + rcap_record(b"hello"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].payload, b"hello")
        self.assertEqual(records[0].record_type, RCAP_TYPE_RAW)
        self.assertEqual(parser.stats.raw_bytes, 5)


class PipelineTests(unittest.TestCase):
    def test_rcap_stream_decodes_embedded_tinyframe(self) -> None:
        with TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "decoded"
            output = OutputWriter(prefix)
            decoder = StreamDecoder("rcap", output, io.StringIO())
            try:
                encoded = rcap_record(target_frame())
                decoder.feed(encoded[:9])
                decoder.feed(encoded[9:])
            finally:
                output.close()
            with prefix.with_suffix(".csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["report"], "target_list")
        self.assertEqual(rows[0]["dop_idx"], "-4")

    def test_training_export_reuses_20hz_schema(self) -> None:
        with TemporaryDirectory() as temporary:
            decoded = Path(temporary) / "decoded.csv"
            decoded.write_text(
                "stream_offset,frame_id,message_type,payload_length,report,record_index,target_num,cluster_id,x_m,y_m,z_m,dop_idx,speed_m_s,area0,area1,area2,area3,detected\n"
                "0,1,0x0A04,24,target_list,0,1,9,1.0,2.0,3.0,-4,,,,,,\n",
                encoding="utf-8",
            )
            training = Path(temporary) / "training.csv"
            count, skipped = _write_training_csv(
                decoded, training, "walking", datetime(2026, 8, 17, 12, 0, 0)
            )
            with training.open(newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual((count, skipped), (1, 0))
        self.assertEqual(rows[0]["activity"], "walking")
        self.assertEqual(rows[0]["timestamp"], "2026-08-17T12:00:00.000000")
        self.assertEqual(rows[0]["cluster_id"], "1")


if __name__ == "__main__":
    unittest.main()
