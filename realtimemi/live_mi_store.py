"""Append live MI updates to a local, long-format CSV event log."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from realtimemi import __version__
from realtimemi.live_mi import CONTEXT_S, HOP_S, WINDOW_S, LiveMIResult


class LiveMIResultStore:
    def __init__(self, output_dir: str | Path, channel_labels: list[str]):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.csv_path = directory / f"mi-live-{run_id}.csv"
        self.json_path = directory / f"mi-live-{run_id}.json"
        self.labels = list(channel_labels)
        settings = {
            "software": "RealTimeMI", "version": __version__, "run_id": run_id,
            "input_type": "growing BESA", "source_sampling_hz": 10_000,
            "channel_labels": self.labels,
            "mi": {"method": "Tort normalized KL", "phase_hz": [3, 4],
                   "amplitude_hz": [80, 250], "window_s": WINDOW_S,
                   "hop_s": HOP_S, "phase_bins": 18,
                   "analytic_fir_taps": 3001, "context_s": CONTEXT_S,
                   "resampling": "10 kHz to 1 kHz, 301-tap Kaiser FIR, 350 Hz cutoff"},
            "csv_semantics": "Rows are updates. For a repeated window/channel, final supersedes provisional.",
        }
        self.json_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(("window_start_s", "window_end_s", "channel_number",
                                         "channel_label", "mi", "status", "elapsed_s",
                                         "recorded_utc"))

    def append(self, result: LiveMIResult):
        stamp = datetime.now(timezone.utc).isoformat()
        status = "final" if result.final_context else "provisional"
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for channel, value in zip(result.channels, result.values):
                writer.writerow((result.start_s, result.end_s, channel + 1,
                                 self.labels[channel],
                                 "nan" if not np.isfinite(value) else f"{float(value):.17g}",
                                 status, f"{result.elapsed_s:.3f}", stamp))
