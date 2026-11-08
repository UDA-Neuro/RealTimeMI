"""Pilot agreement with published automated ripple marks in on003498.

These marks are outputs of another validated detector, not independent
expert-confirmed ground truth. Only pure ripple events are scored here;
FR and FRandR intervals are excluded from the primary comparison.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import firwin, resample_poly

from read_public_ieeg import read_brainvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "RealTimeHFO"))
from realtimehfo.pipeline import analyze_recording  # noqa: E402

DATA = ROOT / "build" / "validation" / "data" / "on003498"
WORK = ROOT / "build" / "validation" / "hfo"
EDGE_S = 0.5
MATCH_MARGIN_S = 0.01


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def pair_names(rows: list[dict]) -> list[str]:
    return sorted({row["trial_type"].split("_", 1)[1] for row in rows
                   if "_" in row["trial_type"]})


def bipolar_data(raw: np.ndarray, labels: list[str], pairs: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(labels)}
    result = np.empty((len(raw), len(pairs)), dtype=np.float32)
    for column, pair in enumerate(pairs):
        match = re.fullmatch(r"([A-Za-z]+)(\d+)-(\d+)", pair)
        if not match:
            raise ValueError(f"Unrecognized electrode pair {pair}")
        left = match.group(1) + match.group(2)
        right = match.group(1) + match.group(3)
        result[:, column] = raw[:, lookup[left]] - raw[:, lookup[right]]
    return result


def overlap(pred: dict, ref: dict, margin: float = MATCH_MARGIN_S) -> bool:
    return (pred["end"] >= ref["start"] - margin and
            pred["start"] <= ref["end"] + margin)


def score(reference: list[dict], predicted: list[dict], ignored: list[dict]) -> dict:
    # Match pure ripple marks first. Only unmatched detections overlapping
    # FR/FRandR intervals are removed from the false-positive denominator.
    matched_predictions = set()
    if reference and predicted:
        cost = np.full((len(reference), len(predicted)), 10.0)
        for i, ref in enumerate(reference):
            for j, pred in enumerate(predicted):
                if ref["pair"] == pred["pair"] and overlap(pred, ref):
                    cost[i, j] = abs(pred["peak"] - (ref["start"] + ref["end"]) / 2)
        row, column = linear_sum_assignment(cost)
        matched_predictions = {int(j) for i, j in zip(row, column) if cost[i, j] < 10}
        tp = len(matched_predictions)
    else:
        tp = 0
    ignored_predictions = sum(any(p["pair"] == x["pair"] and overlap(p, x)
                                  for x in ignored)
                              for j, p in enumerate(predicted) if j not in matched_predictions)
    fp = len(predicted) - tp - ignored_predictions
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"references": len(reference), "predictions_scored": len(predicted) - ignored_predictions,
            "predictions_ignored": ignored_predictions, "tp": tp, "fp": fp,
            "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall else None}


def evaluate(subject: str, run: str) -> dict:
    folder = DATA / f"sub-{subject}" / "ses-interictalsleep" / "ieeg"
    stem = f"sub-{subject}_ses-interictalsleep_run-{run}"
    raw, fs, labels = read_brainvision(folder / f"{stem}_ieeg.vhdr")
    rows = read_tsv(folder / f"{stem}_events.tsv")
    pairs = pair_names(rows)
    derived = bipolar_data(raw, labels, pairs)
    if fs != 2000:
        raise ValueError(f"Unexpected source rate {fs} Hz")
    taps = firwin(301, 350, fs=fs, window=("kaiser", 8.6))
    reduced = resample_poly(derived, 1, 2, axis=0, window=taps).astype(np.float32)
    WORK.mkdir(parents=True, exist_ok=True)
    npz = WORK / f"sub-{subject}_run-{run}_bipolar_1khz.npz"
    if not npz.is_file():
        np.savez(npz, data=reduced, sample_rate_hz=1000, channel_labels=np.array(pairs))
    duration = len(reduced) / 1000
    result = analyze_recording(npz, WORK / f"sub-{subject}_run-{run}-result",
                               duration_s=duration)
    with Path(result["csv_path"]).open(newline="", encoding="utf-8") as stream:
        detected = list(csv.DictReader(stream))
    predictions = [{"pair": row["label"], "start": float(row["start_s"]),
                    "peak": float(row["peak_s"]), "end": float(row["end_s"])}
                   for row in detected if EDGE_S <= float(row["peak_s"]) < duration - EDGE_S]
    reference = []
    ignored = []
    for row in rows:
        kind, pair = row["trial_type"].split("_", 1)
        start = float(row["onset"])
        end = start + float(row["duration"])
        if start < EDGE_S or end > duration - EDGE_S:
            continue
        event = {"pair": pair, "start": start, "end": end}
        (reference if kind == "ripple" else ignored).append(event)
    metrics = score(reference, predictions, ignored)
    return {"dataset": "on003498", "version": "v1.0.0", "subject": subject,
            "run": run, "source_sampling_hz": fs, "analysis_sampling_hz": 1000,
            "source_channels": len(labels), "bipolar_pairs": len(pairs),
            "duration_s": duration, "raw_detector_events": len(detected),
            "matching": "same pair, interval overlap with 10 ms margin; one-to-one",
            "edge_exclusion_s": EDGE_S,
            "reference_scope": "published automated pure-ripple marks; FR and FRandR ignored",
            **metrics}


if __name__ == "__main__":
    report = evaluate("01", "01")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "pilot_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
