"""Exercise the file follower with a BESA file that gains a committed block."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realtimemi.viewer import BesaBlockSource, WaveformChunk, WaveformPlayer, append_waveform


def tag(name: bytes, payload: bytes) -> bytes:
    return struct.pack("<4sI", name, len(payload)) + payload


def make_block(samples: int, value: int) -> bytes:
    channels = b"".join(b"\x00" + np.full(samples, value + i, dtype="<i2").tobytes()
                        for i in range(2))
    return tag(b"BDAT", tag(b"DATT", struct.pack("<I", 0x11)) +
               tag(b"DATS", struct.pack("<I", samples)) + tag(b"DATA", channels))


def commit_snapshot(path: Path, block_offsets: list[int], samples: int):
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        fmi_offset = stream.tell()
        stream.write(tag(b"BFMI", b"\x00" * 8 +
                         tag(b"SAMP", struct.pack("<d", 10_000.0)) +
                         tag(b"SAMT", struct.pack("<Q", len(block_offsets) * samples))))
        table_offset = stream.tell()
        entries = b"".join(tag(b"TAGE", struct.pack("<4sQI", b"BDAT", offset, samples))
                           for offset in block_offsets)
        stream.write(tag(b"BTAG", b"\x00" * 8 + entries))
        stream.seek(0)
        header = tag(b"BCF1", tag(b"OFFM", struct.pack("<Q", fmi_offset)) +
                     tag(b"OFBI", struct.pack("<Q", 64)) +
                     tag(b"OFTL", struct.pack("<Q", table_offset)))
        stream.write(header)


class GrowingBesaTest(unittest.TestCase):
    def test_player_scrolls_across_block_boundaries(self):
        player = WaveformPlayer(window_s=10)
        for index in range(3):
            block = WaveformChunk(np.full((5000, 1), index, dtype=np.float32),
                                  ["A1"], index * 5, (index + 1) * 5, 0)
            player.push(block)
            self.assertTrue(player.advance(5))
        visible = player.visible()
        self.assertEqual((visible.start_s, visible.end_s, visible.data.shape),
                         (5, 15, (10_000, 1)))
        self.assertEqual(float(visible.data[0, 0]), 1)
        self.assertEqual(float(visible.data[-1, 0]), 2)

    def test_follow_replays_latest_then_reads_new_commit(self):
        scratch = Path(__file__).resolve().parents[1] / "build" / "test-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "synthetic.besa"
            labels = tag(b"CHNR", struct.pack("<H", 2))
            for index, name in enumerate(("A1", "A2")):
                labels += tag(b"CHLA", struct.pack("<H", index) + name.encode("utf-16le"))
            with path.open("wb") as stream:
                stream.write(b"\x00" * 64)
                stream.write(tag(b"BCAL", b"\x00" * 8 + labels))
                first_offset = stream.tell()
                stream.write(make_block(10_000, 100))
            commit_snapshot(path, [first_offset], 10_000)
            source = BesaBlockSource(path)
            first = source.poll()
            self.assertEqual((first.start_s, first.end_s), (0, 1))
            self.assertEqual(first.labels, ["A1", "A2"])
            self.assertEqual(first.data.shape, (1000, 2))
            self.assertIsNone(source.poll())
            with path.open("ab") as stream:
                second_offset = stream.tell()
                stream.write(make_block(10_000, 200))
            commit_snapshot(path, [first_offset, second_offset], 10_000)
            second = source.poll()
            self.assertEqual((second.start_s, second.end_s), (1, 2))
            self.assertGreater(float(second.data[500, 0]), float(first.data[500, 0]))
            rolling = append_waveform(first, second)
            self.assertEqual((rolling.start_s, rolling.end_s, rolling.data.shape),
                             (0, 2, (2000, 2)))
            player = WaveformPlayer()
            player.push(first)
            self.assertTrue(player.advance(0.25))
            self.assertAlmostEqual(player.cursor_s, 0.25)
            player.push(second)
            self.assertAlmostEqual(player.cursor_s, 0.25)
            self.assertTrue(player.advance(0.9))
            self.assertAlmostEqual(player.cursor_s, 1.15)
            self.assertEqual(player.visible().data.shape, (1150, 2))
            self.assertTrue(player.advance(2.0))
            self.assertTrue(player.caught_up)
            self.assertFalse(player.advance(0.1))
            self.assertIsNone(source.poll())


if __name__ == "__main__":
    unittest.main()
