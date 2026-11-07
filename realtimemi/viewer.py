"""Read newly committed BESA blocks for the waveform viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from offline.besa_reader import channel_labels, metadata, read_segment


@dataclass
class WaveformChunk:
    data: np.ndarray  # 1 kHz, samples by selected channels
    labels: list[str]
    start_s: float
    end_s: float
    source_mtime: float
    skipped_blocks: int = 0


def append_waveform(previous: WaveformChunk | None, current: WaveformChunk,
                    seconds: int = 10) -> WaveformChunk:
    """Keep a bounded continuous trace for the rolling display."""
    if (previous is None or previous.labels != current.labels or current.skipped_blocks or
            not np.isclose(previous.end_s, current.start_s, atol=1e-6)):
        return current
    values = np.concatenate((previous.data, current.data), axis=0)[-seconds * 1000:]
    return WaveformChunk(values, current.labels, current.end_s - len(values) / 1000,
                         current.end_s, current.source_mtime)


class BesaBlockSource:
    """Follow committed blocks or replay a complete BESA file at recording speed.

    File-follow requires the acquisition software to publish updated BFMI/BTAG
    metadata while writing. Incomplete writes are retried on the next poll.
    """

    def __init__(self, path: str | Path, *, replay: bool = False):
        self.path = Path(path)
        self.replay = replay
        self.next_index: int | None = 0 if replay else None
        self.next_due = 0.0
        self.last_start: int | None = None
        self.labels: list[str] = []
        self.rate = 0.0
        self.channels = 0

    def poll(self, first_channel: int = 0, count: int = 8) -> WaveformChunk | None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self.path.open("rb") as stream:
            rate, total, channels, blocks = metadata(stream)
            labels = channel_labels(stream)
        if rate != 10_000:
            raise ValueError(f"Viewer supports 10 kHz BESA input; file is {rate:g} Hz")
        if not 0 <= first_channel < channels or count < 1:
            raise ValueError("Selected channel is outside the recording")
        self.rate, self.channels, self.labels = rate, channels, labels
        if not blocks:
            return None

        now = time.monotonic()
        if self.replay:
            if now < self.next_due or self.next_index is None or self.next_index >= len(blocks):
                return None
            index = self.next_index
            skipped = 0
        else:
            # Show the latest complete block immediately when joining a recording.
            if self.next_index is None:
                self.next_index = len(blocks) - 1
            if self.next_index >= len(blocks):
                if self.last_start != first_channel:
                    self.next_index = len(blocks) - 1
                else:
                    return None
            index = len(blocks) - 1 if len(blocks) - self.next_index > 2 else self.next_index
            skipped = index - self.next_index

        start_sample = sum(n for _, n in blocks[:index])
        block_samples = blocks[index][1]
        if start_sample + block_samples > total:
            return None
        # read_segment validates BDAT payload and the current tag table again;
        # an in-progress writer update therefore cannot produce a partial trace.
        raw, _ = read_segment(self.path, start_sample / rate, block_samples / rate)
        right = min(first_channel + count, channels)
        display = resample_poly(raw[:, first_channel:right], 1, 10, axis=0).astype(np.float32)
        self.next_index = index + 1
        self.last_start = first_channel
        if self.replay:
            self.next_due = now + block_samples / rate
        return WaveformChunk(display, labels[first_channel:right], start_sample / rate,
                             (start_sample + block_samples) / rate,
                             self.path.stat().st_mtime, skipped)
