"""GPU MI for a 30-second window in a growing BESA recording."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample_poly

from offline.besa_reader import metadata, read_segment
from realtimemi.mi import modulation_index_gpu

WINDOW_S = 30
HOP_S = 15
CONTEXT_S = 1.6  # 1.5 s analytic FIR radius plus resampling margin


@dataclass
class LiveMIResult:
    start_s: int
    end_s: int
    channels: tuple[int, ...]
    values: np.ndarray
    final_context: bool
    elapsed_s: float


def latest_window_start(available_end_s: float) -> int | None:
    """Latest 30 s / 15 s hop window fully inside the committed recording."""
    if available_end_s < WINDOW_S:
        return None
    return max(0, math.floor((available_end_s - WINDOW_S) / HOP_S) * HOP_S)


class LiveMIPlanner:
    """Queue each 30 s window and revisit provisional edge results once."""

    def __init__(self, replay: bool = False):
        self.replay = replay
        self.next_start: int | None = None
        self.pending: deque[int] = deque()
        self.queued: set[int] = set()
        self.inflight: set[int] = set()
        self.provisional: set[int] = set()
        self.finalized: set[int] = set()
        self.retry: set[int] = set()
        self.retry_after: dict[int, float] = {}

    def _enqueue(self, start: int):
        if start not in self.queued and start not in self.inflight and start not in self.finalized:
            self.pending.append(start)
            self.queued.add(start)

    def observe(self, display_end_s: float, committed_end_s: float):
        threshold = display_end_s if self.replay else committed_end_s
        if self.next_start is None:
            self.next_start = 0 if self.replay else (latest_window_start(threshold) or 0)
        for start in sorted(self.retry | self.provisional):
            if ((start in self.retry and time.monotonic() >= self.retry_after.get(start, 0)) or
                    (start in self.provisional and committed_end_s >= start + WINDOW_S + CONTEXT_S)):
                self._enqueue(start)
        while self.next_start + WINDOW_S <= threshold + 1e-6:
            self._enqueue(self.next_start)
            self.next_start += HOP_S

    def pop(self) -> int | None:
        if not self.pending:
            return None
        start = self.pending.popleft()
        self.queued.remove(start)
        self.inflight.add(start)
        self.retry.discard(start)
        self.retry_after.pop(start, None)
        return start

    def complete(self, result: LiveMIResult):
        self.inflight.discard(result.start_s)
        self.retry.discard(result.start_s)
        self.retry_after.pop(result.start_s, None)
        if result.final_context:
            self.provisional.discard(result.start_s)
            self.finalized.add(result.start_s)
        else:
            self.provisional.add(result.start_s)

    def failed(self, start: int):
        self.inflight.discard(start)
        self.retry.add(start)
        self.retry_after[start] = time.monotonic() + 2


def compute_live_mi_window(path: str | Path, start_s: int,
                           channels: tuple[int, ...]) -> LiveMIResult:
    """Compute one window using available context and the offline MI estimator.

    A result without right context is provisional. Recompute it when another
    1.6 seconds has been committed so that it matches offline filtering.
    """
    started = time.perf_counter()
    if start_s < 0 or start_s % HOP_S:
        raise ValueError("MI window start must be a nonnegative 15-second multiple")
    with Path(path).open("rb") as stream:
        rate, total, channel_count, _ = metadata(stream)
    if rate != 10_000:
        raise ValueError("Live MI requires 10 kHz BESA input")
    if not channels or len(set(channels)) != len(channels) or any(
            channel < 0 or channel >= channel_count for channel in channels):
        raise ValueError("Invalid MI channel selection")
    end_s = start_s + WINDOW_S
    if total < end_s * rate:
        raise ValueError("MI window is not yet complete")
    committed_end_s = total / rate
    read_start = max(0.0, start_s - CONTEXT_S)
    read_end = min(committed_end_s, end_s + CONTEXT_S)
    raw, source_rate = read_segment(path, read_start, read_end - read_start, channels)
    if source_rate != rate:
        raise RuntimeError("BESA sampling rate changed during read")
    taps = firwin(301, 350, fs=10_000, window=("kaiser", 8.6))
    reduced = resample_poly(raw, 1, 10, axis=0, window=taps).astype(np.float32)
    window_start = round((start_s - read_start) * 1000)
    values, _ = modulation_index_gpu(reduced, 1000,
                                     window_start_samples=np.array([window_start]))
    return LiveMIResult(start_s, end_s, channels, values[0],
                        committed_end_s >= end_s + CONTEXT_S,
                        time.perf_counter() - started)
