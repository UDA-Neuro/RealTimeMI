"""Diagnose pilot disagreement without changing either detector or benchmark.

This is an exploratory, dataset-specific audit. The HFO marks are automated
outputs of another algorithm; an unmatched spike is not a proven false alarm.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, find_peaks, sosfiltfilt

from evaluate_hfo import EDGE_S, MATCH_MARGIN_S, bipolar_data, pair_names, read_tsv, score as score_hfo
from evaluate_spike import CLUSTER_S, TOLERANCE_S
from read_public_ieeg import read_brainvision


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "validation" / "diagnosis.json"
sys.path.insert(0, str(ROOT / "build" / "RealTimeHFO"))
from realtimehfo.detector import detect as detect_hfo  # noqa: E402


def percentile(values: list[float]) -> dict:
    return {str(p): round(float(np.percentile(values, p)), 3) for p in (10, 50, 90)} if values else {}


def hfo_audit() -> dict:
    work = ROOT / "build" / "validation" / "hfo"
    with np.load(work / "sub-01_run-01_bipolar_1khz.npz", allow_pickle=False) as archive:
        data = archive["data"]
        pairs = [str(x) for x in archive["channel_labels"]]
    rows = read_tsv(ROOT / "build" / "validation" / "data" / "on003498" /
                    "sub-01" / "ses-interictalsleep" / "ieeg" /
                    "sub-01_ses-interictalsleep_run-01_events.tsv")
    references = defaultdict(list)
    for row in rows:
        kind, pair = row["trial_type"].split("_", 1)
        start = float(row["onset"])
        end = start + float(row["duration"])
        if kind == "ripple" and EDGE_S <= start and end <= len(data) / 1000 - EDGE_S:
            references[pair].append((start, end))
    sos = butter(4, (80, 250), btype="bandpass", fs=1000, output="sos")
    stages = Counter()
    ratios = []
    reference_durations_ms = []
    local_thresholds = []
    for column, pair in enumerate(pairs):
        for core_start in range(0, len(data) // 1000, 5):
            core_end = core_start + 5
            selected = [(start, end) for start, end in references[pair]
                        if core_start <= (start + end) / 2 < core_end]
            if not selected:
                continue
            read_start = max(0, core_start - 0.5)
            read_end = min(len(data) / 1000, core_end + 0.5)
            raw = data[round(read_start * 1000):round(read_end * 1000), column].astype(np.float64)
            filtered = sosfiltfilt(sos, raw)
            rms = np.sqrt(np.maximum(0, uniform_filter1d(filtered**2, size=5, mode="nearest")))
            threshold = float(np.mean(rms) + 5 * np.std(rms))
            local_thresholds.append(threshold)
            boundaries = np.flatnonzero(np.diff(np.r_[False, rms > threshold, False])).reshape(-1, 2)
            spans = [(int(a), int(b)) for a, b in boundaries]
            merged = []
            for left, right in spans:
                if merged and left - merged[-1][1] <= 10:
                    merged[-1][1] = right
                else:
                    merged.append([left, right])
            duration_ok = [(left, right) for left, right in merged if 6 <= right - left <= 200]
            cycles_ok = [(left, right) for left, right in duration_ok
                         if len(find_peaks(filtered[left:right], distance=4)[0]) >= 4]

            def intersects(spans_to_check, start, end):
                left = round((start - MATCH_MARGIN_S - read_start) * 1000)
                right = round((end + MATCH_MARGIN_S - read_start) * 1000)
                return any(a <= right and b >= left for a, b in spans_to_check)

            for start, end in selected:
                stages["references"] += 1
                reference_durations_ms.append((end - start) * 1000)
                left = max(0, round((start - MATCH_MARGIN_S - read_start) * 1000))
                right = min(len(rms), round((end + MATCH_MARGIN_S - read_start) * 1000) + 1)
                ratios.append(float(np.max(rms[left:right]) / threshold))
                for name, stage in (("above_threshold", spans), ("after_merge", merged),
                                    ("duration_ok", duration_ok), ("four_cycles_ok", cycles_ok)):
                    stages[name] += intersects(stage, start, end)
    return {"stages": dict(stages), "max_rms_to_threshold_percentiles": percentile(ratios),
            "fraction_reference_max_below_threshold": round(float(np.mean(np.array(ratios) <= 1)), 4),
            "reference_duration_ms_percentiles": percentile(reference_durations_ms),
            "local_threshold_percentiles": percentile(local_thresholds),
            "note": "Each reference assigned by midpoint to a 5 s core; intersections use 10 ms margin."
            }


def hfo_native_rate_check() -> dict:
    folder = ROOT / "build" / "validation" / "data" / "on003498" / "sub-01" / "ses-interictalsleep" / "ieeg"
    stem = "sub-01_ses-interictalsleep_run-01"
    raw, fs, labels = read_brainvision(folder / f"{stem}_ieeg.vhdr")
    rows = read_tsv(folder / f"{stem}_events.tsv")
    pairs = pair_names(rows)
    data = bipolar_data(raw, labels, pairs)
    duration = len(data) / fs
    references = []
    ignored = []
    for row in rows:
        kind, pair = row["trial_type"].split("_", 1)
        start = float(row["onset"])
        end = start + float(row["duration"])
        if EDGE_S <= start and end <= duration - EDGE_S:
            (references if kind == "ripple" else ignored).append({"pair": pair, "start": start, "end": end})
    predictions = []
    for core_start in range(0, int(duration), 5):
        core_end = min(core_start + 5, duration)
        read_start = max(0, core_start - 0.5)
        read_end = min(duration, core_end + 0.5)
        chunk = data[round(read_start * fs):round(read_end * fs)]
        for event in detect_hfo(chunk, fs, read_start, pairs):
            peak = event["peak_s"]
            if core_start <= peak < core_end and EDGE_S <= peak < duration - EDGE_S:
                predictions.append({"pair": event["label"], "start": event["start_s"],
                                    "peak": peak, "end": event["end_s"]})
    return {"sampling_hz": fs, **score_hfo(references, predictions, ignored)}


def spike_audit() -> dict:
    work = ROOT / "build" / "validation" / "spike"
    result = {}
    for subject in ("01", "02", "03"):
        predictions_file = max((work / f"sub-{subject}-result").glob("spike-*.csv"), key=lambda p: p.stat().st_mtime)
        with predictions_file.open(newline="", encoding="utf-8") as stream:
            raw = list(csv.DictReader(stream))
        references_file = ROOT / "build" / "validation" / "data" / "nm000360" / f"sub-{subject}" / "ieeg" / f"sub-{subject}_task-sleep_events.tsv"
        references = [{"time": float(r["onset"]), "side": "R" if r["trial_type"].startswith("Rt") else "L"}
                      for r in read_tsv(references_file) if r["trial_type"].startswith(("Rt", "Lt"))]
        interpretation_file = ROOT / "build" / "validation" / "data" / "nm000360" / "derivatives" / f"sub-{subject}_task-sleep_events_interpretation.tsv"
        interpreted = [{"time": float(r["time_in_sec"]), "channels": set(r["chans"].split())}
                       for r in read_tsv(interpretation_file)]
        channel_reference = [(r["time"], channel) for r in interpreted for channel in r["channels"]]
        channel_cost = np.full((len(channel_reference), len(raw)), 10.0)
        event_cost = np.full((len(interpreted), len(raw)), 10.0)
        for i, (time, channel) in enumerate(channel_reference):
            for j, pred in enumerate(raw):
                difference = abs(time - float(pred["peak_s"]))
                if channel == pred["label"] and difference <= TOLERANCE_S:
                    channel_cost[i, j] = difference
        for i, ref in enumerate(interpreted):
            for j, pred in enumerate(raw):
                difference = abs(ref["time"] - float(pred["peak_s"]))
                if pred["label"] in ref["channels"] and difference <= TOLERANCE_S:
                    event_cost[i, j] = difference
        ci, cj = linear_sum_assignment(channel_cost)
        ei, ej = linear_sum_assignment(event_cost)
        channel_matches = int(np.sum(channel_cost[ci, cj] <= TOLERANCE_S))
        event_matches = int(np.sum(event_cost[ei, ej] <= TOLERANCE_S))
        groups = []
        for side in ("L", "R"):
            rows = sorted((r for r in raw if r["label"].upper().startswith(side)), key=lambda r: float(r["peak_s"]))
            pending = []
            for row in rows:
                if pending and float(row["peak_s"]) - float(pending[0]["peak_s"]) > CLUSTER_S:
                    groups.append({"side": side, "time": float(np.median([float(p["peak_s"]) for p in pending])),
                                   "channels": len(pending), "ratio": max(float(p["score"]) / float(p["threshold"]) for p in pending)})
                    pending = []
                pending.append(row)
            if pending:
                groups.append({"side": side, "time": float(np.median([float(p["peak_s"]) for p in pending])),
                               "channels": len(pending), "ratio": max(float(p["score"]) / float(p["threshold"]) for p in pending)})
        cost = np.full((len(references), len(groups)), 10.0)
        for i, ref in enumerate(references):
            for j, pred in enumerate(groups):
                if ref["side"] == pred["side"]:
                    distance = abs(ref["time"] - pred["time"])
                    if distance <= TOLERANCE_S:
                        cost[i, j] = distance
        ri, pi = linear_sum_assignment(cost)
        matched = {int(j) for i, j in zip(ri, pi) if cost[i, j] <= TOLERANCE_S}
        unmatched = [pred for j, pred in enumerate(groups) if j not in matched]
        match = [pred for j, pred in enumerate(groups) if j in matched]
        nearest = [min((abs(p["time"] - r["time"]) for r in references if r["side"] == p["side"]),
                       default=float("inf")) for p in unmatched]
        result[subject] = {
            "reference_by_side": dict(Counter(r["side"] for r in references)),
            "cluster_by_side": dict(Counter(p["side"] for p in groups)),
            "unmatched_by_side": dict(Counter(p["side"] for p in unmatched)),
            "unmatched_nearest_same_side_s": {
                "0.1_to_0.5": sum(0.1 < d <= 0.5 for d in nearest),
                "0.5_to_2": sum(0.5 < d <= 2 for d in nearest),
                "over_2_or_no_reference": sum(d > 2 for d in nearest)},
            "score_to_threshold_percentiles": {"matched": percentile([p["ratio"] for p in match]),
                                                "unmatched": percentile([p["ratio"] for p in unmatched])},
            "channels_per_cluster_percentiles": {"matched": percentile([p["channels"] for p in match]),
                                                 "unmatched": percentile([p["channels"] for p in unmatched])},
            "matched": len(match), "unmatched": len(unmatched)}
        result[subject]["channel_specific"] = {
            "interpreted_events": len(interpreted), "event_matches_with_listed_channel": event_matches,
            "listed_event_channel_pairs": len(channel_reference), "matched_event_channel_pairs": channel_matches,
            "raw_channel_predictions": len(raw),
            "note": "Same channel and within 100 ms; pair score is one-to-one."
        }
    return result


if __name__ == "__main__":
    report = {"hfo": hfo_audit(), "hfo_native_rate": hfo_native_rate_check(), "spike": spike_audit()}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

