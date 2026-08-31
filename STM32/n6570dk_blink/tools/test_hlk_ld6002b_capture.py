import struct
import time
import unittest
import tempfile
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from hlk_ld6002b_capture import (  # noqa: E402
    OutputWriter,
    TinyFrameParser,
    decode_report,
    process_chunks,
    tf_checksum,
)


def frame(frame_id: int, message_type: int, payload: bytes) -> bytes:
    header_without_checksum = (
        b"\x01"
        + frame_id.to_bytes(2, "big")
        + len(payload).to_bytes(2, "big")
        + message_type.to_bytes(2, "big")
    )
    return header_without_checksum + bytes([tf_checksum(header_without_checksum)]) + payload + bytes([tf_checksum(payload)])


def target_frame(frame_id: int = 1) -> bytes:
    payload = struct.pack("<i", 1) + struct.pack("<fffii", 1.25, -2.5, 0.5, -17, 9)
    return frame(frame_id, 0x0A04, payload)


class TinyFrameParserTests(unittest.TestCase):
    def test_valid_frame_split_across_chunks_decodes_target(self) -> None:
        encoded = target_frame()
        parser = TinyFrameParser()
        frames = parser.feed(encoded[:5])
        self.assertEqual(frames, [])
        frames = parser.feed(encoded[5:13])
        self.assertEqual(frames, [])
        frames = parser.feed(encoded[13:])
        self.assertEqual(len(frames), 1)
        decoded = decode_report(frames[0])
        self.assertEqual(decoded["report"], "target_list")
        self.assertEqual(decoded["target_num"], 1)
        self.assertAlmostEqual(decoded["targets"][0]["x_m"], 1.25)
        self.assertEqual(decoded["targets"][0]["dop_idx"], -17)
        self.assertEqual(parser.stats.valid_frames, 1)

    def test_truncated_stream_times_out_and_recovers(self) -> None:
        encoded = target_frame()
        parser = TinyFrameParser(frame_timeout_s=0.01)
        parser.feed(encoded[:-3])
        self.assertTrue(parser.expire_partial(time.monotonic() + 1.0))
        self.assertEqual(parser.stats.timeouts, 1)
        frames = parser.feed(encoded)
        self.assertEqual(len(frames), 1)

    def test_corrupt_checksum_is_rejected_then_next_frame_recovers(self) -> None:
        corrupt = bytearray(target_frame(1))
        corrupt[-1] ^= 0x01
        parser = TinyFrameParser()
        frames = parser.feed(bytes(corrupt) + target_frame(2))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_id, 2)
        self.assertEqual(parser.stats.invalid_checksums, 1)

    def test_concatenated_report_types_decode(self) -> None:
        cloud_payload = struct.pack("<iiffff", 1, 3, 1.0, 2.0, 3.0, -0.75)
        area_payload = struct.pack("<IIII", 0, 1, 0, 0)
        parser = TinyFrameParser()
        frames = parser.feed(target_frame(1) + frame(2, 0x0A08, cloud_payload) + frame(3, 0x0A0A, area_payload))
        decoded = [decode_report(item) for item in frames]
        self.assertEqual([item["report"] for item in decoded], ["target_list", "point_cloud", "area_presence"])
        self.assertAlmostEqual(decoded[1]["points"][0]["speed_m_s"], -0.75)
        self.assertEqual(decoded[2]["area_states"], [0, 1, 0, 0])
        self.assertEqual(decoded[2]["detected"], 1)

    def test_oversized_length_is_counted_and_resynchronizes(self) -> None:
        header_without_checksum = b"\x01\x00\x01\x04\x01\x0A\x04"
        oversized = header_without_checksum + bytes([tf_checksum(header_without_checksum)])
        parser = TinyFrameParser()
        frames = parser.feed(oversized + target_frame(2))
        self.assertEqual(len(frames), 1)
        self.assertEqual(parser.stats.oversized_frames, 1)

    def test_decoded_output_is_written_as_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "capture"
            writer = OutputWriter(prefix)
            try:
                process_chunks([target_frame()], TinyFrameParser(), writer, io.StringIO())
            finally:
                writer.close()
            jsonl = prefix.with_suffix(".jsonl").read_text(encoding="utf-8")
            csv_text = prefix.with_suffix(".csv").read_text(encoding="utf-8")
            self.assertIn('"report":"target_list"', jsonl)
            self.assertIn("stream_offset,frame_id,message_type", csv_text)
            self.assertIn("0x0A04", csv_text)


if __name__ == "__main__":
    unittest.main()
