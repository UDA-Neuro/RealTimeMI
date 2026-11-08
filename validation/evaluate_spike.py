"""Pilot event-level comparison on pinned nm000360 subjects.

The dataset marks side and time, not a unique electrode. Predictions from
channels on the same side are clustered before one-to-one matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from read_public_ieeg import read_edf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "RealTimeSpike"))
from realtimespike.pipeline import analyze_recording  # noqa: E402

DATA = ROOT / "build" / "validation" / "data" / "nm000360"
WORK = ROOT / "build" / "validation" / "spike"
CLUSTER_S = 0.05
TOLERANCE_S = 0.1


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def cluster_predictions(rows: list[dict]) -> list[dict]:
    groups = []
    for side in ("L", "R"):
        candidates = sorted((float(row["peak_s"]), row["label"])
                            for row in rows if row["label"].upper().startswith(side))
        pending = []
        for peak, label in candidates:
            if pending and peak - pending[0][0] > CLUSTER_S:
                groups.append({"side": side, "time": float(np.median([p for p, _ in pending])),
                               "channels": len(pending)})
                pending = []
            pending.append((peak, label))
        if pending:
            groups.append({"side": side, "time": float(np.median([p for p, _ in pending])),
                           "channels": len(pending)})
    return sorted(groups, key=lambda row: row["time"])


def score(reference: list[dict], predicted: list[dict]) -> dict:
    if not reference or not predicted:
        tp = 0
    else:
        cost = np.full((len(reference), len(predicted)), 10.0)
        for i, event in enumerate(reference):
            for j, candidate in enumerate(predicted):
                if event["side"] == candidate["side"]:
                    difference = abs(event["time"] - candidate["time"])
                    if difference <= TOLERANCE_S:
                        cost[i, j] = difference
        row, column = linear_sum_assignment(cost)
        tp = int(np.sum(cost[row, column] <= TOLERANCE_S))
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    sensitivity = tp / (tp + fn) if tp + fn else None
    return {"references": len(reference), "predictions": len(predicted),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "sensitivity": sensitivity,
            "f1": 2 * precision * sensitivity / (precision + sensitivity)
            if precision is not None and sensitivity is not None and precision + sensitivity else None}


def evaluate(subject: str) -> dict:
    directory = DATA / f"sub-{subject}" / "ieeg"
    edf = directory / f"sub-{subject}_task-sleep_ieeg.edf"
    tsv = directory / f"sub-{subject}_task-sleep_events.tsv"
    values, fs, labels = read_edf(edf)
    if fs != 1000:
        raise ValueError(f"Unexpected rate {fs} Hz in {edf}")
    WORK.mkdir(parents=True, exist_ok=True)
    npz = WORK / f"sub-{subject}.npz"
    if not npz.is_file():
        np.savez(npz, data=values, sample_rate_hz=fs, channel_labels=np.array(labels))
    result = analyze_recording(npz, WORK / f"sub-{subject}-result",
                               duration_s=len(values) / fs)
    with Path(result["csv_path"]).open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    annotations = read_tsv(tsv)
    reference = [{"side": "R" if row["trial_type"].startswith("Rt") else "L",
                  "time": float(row["onset"])} for row in annotations
                 if row["trial_type"].startswith(("Rt", "Lt"))]
    grouped = cluster_predictions(predictions)
    metrics = score(reference, grouped)
    return {"dataset": "nm000360", "version": "v1.0.2", "subject": subject,
            "sampling_hz": fs, "channels": len(labels),
            "duration_s": len(values) / fs, "raw_detector_events": len(predictions),
            "cluster_s": CLUSTER_S, "match_tolerance_s": TOLERANCE_S,
            "matching": "same side, one-to-one closest absolute time",
            **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+", help="Two-digit subject IDs")
    args = parser.parse_args()
    reports = [evaluate(subject) for subject in args.subjects]
    destination = WORK / "pilot_metrics.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
