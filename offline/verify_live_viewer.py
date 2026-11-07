"""Exercise the real GUI against a paced, synthetic growing BESA file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.live_besa_simulator import SyntheticBesaWriter
from realtimemi.viewer_gui import EEGViewer


def main():
    parser = argparse.ArgumentParser(description="Paced synthetic BESA → GUI → GPU MI integration test")
    parser.add_argument("--block-interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.block_interval <= 0:
        parser.error("block interval must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(__file__).resolve().parents[1] / "build" / "live-sim-test" / stamp
    directory.mkdir(parents=True, exist_ok=True)
    writer = SyntheticBesaWriter(directory / "synthetic_live.besa")
    commits: list[dict] = []
    first_start, first_end = writer.append_block()
    started = time.monotonic()
    commits.append({"data_end_s": first_end, "wall_s": 0.0})
    failures: list[str] = []

    def produce():
        try:
            for index in range(1, 10):
                due = started + index * args.block_interval
                time.sleep(max(0, due - time.monotonic()))
                _, end = writer.append_block()
                commits.append({"data_end_s": end,
                                "wall_s": round(time.monotonic() - started, 3)})
                print(f"Committed synthetic {end:g} s at wall {commits[-1]['wall_s']:.2f} s", flush=True)
        except Exception as exc:
            failures.append(f"writer: {type(exc).__name__}: {exc}")

    threading.Thread(target=produce, daemon=True).start()
    root = tk.Tk()
    viewer = EEGViewer(root, directory / "results")
    viewer.path.set(str(writer.path))
    viewer.mode.set("follow")
    viewer.start()
    cursors: list[float] = []
    deadline = started + 10 * args.block_interval + 25

    def inspect():
        if viewer.player.cursor_s is not None:
            cursors.append(viewer.player.cursor_s)
        if failures:
            viewer.close()
            return
        done = (len(commits) == 10 and viewer.mi_store is not None and
                all(0 in viewer.mi_results.get(ch, {}) and
                    15 in viewer.mi_results.get(ch, {}) and
                    viewer.mi_results[ch][0][1] and viewer.mi_results[ch][15][1]
                    for ch in range(8)))
        if done:
            with viewer.mi_store.csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            statuses = {(int(row["window_start_s"]), row["status"]) for row in rows}
            required = {(0, "provisional"), (0, "final"),
                        (15, "provisional"), (15, "final")}
            if not required.issubset(statuses):
                failures.append(f"Missing MI update stages: {required - statuses}")
            if not cursors or max(cursors) - min(cursors) < 20:
                failures.append("Waveform cursor did not advance across several blocks")
            if len(viewer.canvas.find_all()) < 10 or len(viewer.mi_canvas.find_all()) < 5:
                failures.append("Waveform or MI canvas did not render")
            if not viewer.mi_results[0][0][0] > viewer.mi_results[7][0][0]:
                failures.append("Simulated coupled A1 did not exceed uncoupled A8 MI")
            summary = {
                "simulated": True, "block_interval_s": args.block_interval,
                "commits": commits, "elapsed_wall_s": round(time.monotonic() - started, 3),
                "waveform_cursor_first_last_s": [round(min(cursors), 3), round(max(cursors), 3)],
                "mi_csv": str(viewer.mi_store.csv_path.resolve()),
                "mi_csv_rows": len(rows), "mi_stages": sorted([list(item) for item in statuses]),
                "a1_mi_0_30": viewer.mi_results[0][0][0],
                "a8_mi_0_30": viewer.mi_results[7][0][0],
                "failures": failures,
            }
            (directory / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
            viewer.close()
            return
        if time.monotonic() > deadline:
            report = {"commits": commits, "mi_status": viewer.mi_status.get(),
                      "wave_status": viewer.status.get(), "mi_channels": len(viewer.mi_results)}
            (directory / "timeout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            failures.append(f"Live viewer did not complete: {report}")
            viewer.close()
            return
        root.after(250, inspect)

    root.after(250, inspect)
    root.mainloop()
    if failures:
        raise AssertionError(failures)


if __name__ == "__main__":
    main()
