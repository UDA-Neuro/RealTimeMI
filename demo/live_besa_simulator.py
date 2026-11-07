"""Create a synthetic 10 kHz BESA file one committed block at a time.

This is a local development aid. Every waveform is generated mathematically;
no patient recording is used.
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np


def _tag(name: bytes, payload: bytes) -> bytes:
    return struct.pack("<4sI", name, len(payload)) + payload


class SyntheticBesaWriter:
    def __init__(self, path: str | Path, channels: int = 8, block_seconds: int = 5):
        if channels < 2 or block_seconds < 1:
            raise ValueError("Need at least two channels and one-second blocks")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.channels = channels
        self.block_samples = block_seconds * 10_000
        self.offsets: list[int] = []
        labels = _tag(b"CHNR", struct.pack("<H", channels))
        for index in range(channels):
            labels += _tag(b"CHLA", struct.pack("<H", index) +
                           f"A{index + 1}".encode("utf-16le"))
        with self.path.open("xb") as stream:
            stream.write(b"\x00" * 64)
            stream.write(_tag(b"BCAL", b"\x00" * 8 + labels))

    def append_block(self) -> tuple[float, float]:
        index = len(self.offsets)
        sample = index * self.block_samples + np.arange(self.block_samples)
        t = sample / 10_000.0
        phase = 2 * np.pi * 3.5 * t
        high = np.sin(2 * np.pi * 130 * t)
        encoded = bytearray()
        for channel in range(self.channels):
            depth = max(0, 0.9 - 0.15 * channel)
            signal = 1200 * np.sin(phase + 0.1 * channel)
            signal += 700 * (1 + depth * np.cos(phase)) * high
            encoded.extend(b"\x00")
            encoded.extend(np.rint(signal).astype("<i2").tobytes())
        block = _tag(b"BDAT", _tag(b"DATT", struct.pack("<I", 0x11)) +
                     _tag(b"DATS", struct.pack("<I", self.block_samples)) +
                     _tag(b"DATA", bytes(encoded)))
        with self.path.open("r+b") as stream:
            stream.seek(0, 2)
            offset = stream.tell()
            stream.write(block)
            self.offsets.append(offset)
            fmi_offset = stream.tell()
            total = len(self.offsets) * self.block_samples
            stream.write(_tag(b"BFMI", b"\x00" * 8 +
                              _tag(b"SAMP", struct.pack("<d", 10_000.0)) +
                              _tag(b"SAMT", struct.pack("<Q", total))))
            table_offset = stream.tell()
            entries = b"".join(_tag(b"TAGE", struct.pack("<4sQI", b"BDAT", item,
                                                              self.block_samples))
                               for item in self.offsets)
            stream.write(_tag(b"BTAG", b"\x00" * 8 + entries))
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            header = _tag(b"BCF1", _tag(b"OFFM", struct.pack("<Q", fmi_offset)) +
                          _tag(b"OFBI", struct.pack("<Q", 64)) +
                          _tag(b"OFTL", struct.pack("<Q", table_offset)))
            if len(header) > 64:
                raise RuntimeError("Synthetic BESA header exceeds reserved space")
            stream.write(header)
            stream.flush()
            os.fsync(stream.fileno())
        return index * self.block_samples / 10_000, total / 10_000


def main():
    parser = argparse.ArgumentParser(description="Simulate a growing synthetic BESA recording")
    parser.add_argument("--output", required=True, help="New .besa file; existing files are never overwritten")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-interval", type=float, default=5.0,
                        help="Wall-clock seconds between five-second data blocks")
    args = parser.parse_args()
    if args.blocks < 1 or args.block_interval < 0:
        parser.error("blocks must be positive and interval nonnegative")
    writer = SyntheticBesaWriter(args.output)
    for index in range(args.blocks):
        start, end = writer.append_block()
        print(f"Committed synthetic EEG {start:g}–{end:g} s", flush=True)
        if index + 1 < args.blocks:
            time.sleep(args.block_interval)


if __name__ == "__main__":
    main()
