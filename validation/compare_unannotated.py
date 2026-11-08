"""Agreement-only pilot on annotation-free Kuroda interictal sleep iEEG.

This compares independent detector implementations, not clinical accuracy.
Third-party epycom is installed only under ignored build/validation/comparators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import spearmanr

from read_public_ieeg import read_edf


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build" / "validation" / "data" / "on006107"
OUT = ROOT / "build" / "validation" / "unannotated"
VENDOR = ROOT / "build" / "validation" / "comparators" / "epycom_pkg"
sys.path.insert(0, str(ROOT / "build" / "RealTimeHFO"))
sys.path.insert(0, str(ROOT / "build" / "RealTimeSpike"))
from realtimehfo.detector import detect as detect_hfo  # noqa: E402
from realtimespike.detector import detect as detect_spike  # noqa: E402


def load_epycom():
    """Load two untouched epycom modules without importing optional HFO/numba modules.

    epycom 0.3's umbrella __init__ imports its unrelated C/S HFO detector,
    which requires numba. These two detector modules have no numba dependency.
    """
    if not (VENDOR / "epycom").is_dir():
        raise RuntimeError("Install epycom==0.3 with: python -m pip install --no-deps "
                           "--target build/validation/comparators/epycom_pkg epycom==0.3")
    sys.path.insert(0, str(VENDOR))
    importlib.import_module("epycom")
    root = VENDOR / "epycom" / "event_detection"
    for name, path in (("epycom.event_detection", root),
                       ("epycom.event_detection.hfo", root / "hfo"),
                       ("epycom.event_detection.spike", root / "spike")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    rms = importlib.import_module("epycom.event_detection.hfo.rms_detector")
    janca = importlib.import_module("epycom.event_detection.spike.janca_detector")
    # epycom 0.3 hard-codes a 50 Hz notch despite exposing line_freq. Disable
    # that optional preprocessing so both spike detectors receive unnotched
    # input. The 10-60 Hz detector band remains active in both implementations.
    janca._filt50Hz = lambda data, fs, bandwidth: data
    return rms.detect_hfo_rms, janca.detect_spikes_janca


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def revision(folder: Path) -> str:
    result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def score_events(a: list[dict], b: list[dict], margin_s: float, labels: list[str],
                 *, point_mode: bool = False) -> dict:
    """One-to-one same-channel interval overlap; no reference is a truth label."""
    by_channel_a = {}
    by_channel_b = {}
    for item in a:
        by_channel_a.setdefault(item["label"], []).append(item)
    for item in b:
        by_channel_b.setdefault(item["label"], []).append(item)
    pairs = []
    for label in sorted(set(by_channel_a) | set(by_channel_b)):
        left = by_channel_a.get(label, [])
        right = by_channel_b.get(label, [])
        if not left or not right:
            continue
        cost = np.full((len(left), len(right)), 1e6)
        for i, x in enumerate(left):
            for j, y in enumerate(right):
                acceptable = (abs(x["peak_s"] - y["peak_s"]) <= margin_s if point_mode else
                              x["end_s"] + margin_s >= y["start_s"] and
                              y["end_s"] + margin_s >= x["start_s"])
                if acceptable:
                    cost[i, j] = abs(x["peak_s"] - y["peak_s"])
        ri, rj = linear_sum_assignment(cost)
        pairs.extend((left[i], right[j]) for i, j in zip(ri, rj) if cost[i, j] < 1e6)
    n = len(pairs)
    counts_a = np.array([len(by_channel_a.get(label, [])) for label in labels])
    counts_b = np.array([len(by_channel_b.get(label, [])) for label in labels])
    rho = float(spearmanr(counts_a, counts_b).statistic) if (len(labels) >= 3 and
          np.ptp(counts_a) > 0 and np.ptp(counts_b) > 0) else None
    return {"detector_a_events": len(a), "detector_b_events": len(b), "matched_pairs": n,
            "a_matched_fraction": n / len(a) if a else None,
            "b_matched_fraction": n / len(b) if b else None,
            "event_jaccard": n / (len(a) + len(b) - n) if a or b else None,
            "channel_count_spearman_rho": rho,
            "median_peak_delta_ms": float(np.median([abs(x["peak_s"] - y["peak_s"]) * 1000 for x, y in pairs])) if pairs else None}


def epycom_hfo(filtered: np.ndarray, fs: int, label: str, function,
               *, package_window_size: int, center_shift: int) -> list[dict]:
    # epycom runs its own RMS and mean+5SD threshold on the same filtered input.
    # epycom reports the end of the final window, whereas our detector
    # reports the end of the last above-threshold RMS sample.
    # Convert to the same convention before applying shared morphology rules.
    spans = [[int(a) + center_shift, max(int(a) + center_shift + 1,
                                        int(b) - package_window_size + center_shift)]
             for a, b in function(filtered, fs=fs, threshold=5,
                                  window_size=package_window_size,
                                  window_overlap=1 / package_window_size)]
    merged = []
    for left, right in spans:
        if merged and left - merged[-1][1] <= 10:
            merged[-1][1] = right
        else:
            merged.append([left, right])
    events = []
    for left, right in merged:
        if not 6 <= right - left <= 200:
            continue
        positive = find_peaks(filtered[left:right], distance=4)[0]
        if len(positive) < 4:
            continue
        peak = left + int(np.argmax(np.abs(filtered[left:right])))
        events.append({"label": label, "start_s": left / fs, "peak_s": peak / fs,
                       "end_s": right / fs})
    return events


def convert_janca(result, label: str, fs: int) -> list[dict]:
    # epycom 0.3 returns (event_peak_sample, envelope_amplitude, raw_amplitude).
    if result is None or len(result) == 0:
        return []
    if len(result[0]) != 3:
        raise ValueError(f"Unexpected Janca output: {result[0]}")
    return [{"label": label, "start_s": float(row[0]) / fs,
             "peak_s": float(row[0]) / fs, "end_s": float(row[0]) / fs} for row in result]


def evaluate(subject: str, seconds: int, max_channels: int) -> dict:
    files = sorted((DATA / f"sub-{subject}").glob("**/*_ieeg.edf"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one EDF for sub-{subject}, got {files}")
    path = files[0]
    source_hash = digest(path)
    manifest = json.loads((ROOT / "build" / "validation" / "manifests" / "on006107.json").read_text(encoding="utf-8"))
    relative = path.relative_to(DATA).as_posix()
    pinned = next((item for item in manifest if item["path"] == relative), None)
    if pinned is None or pinned["checksum_algorithm"] != "sha256" or pinned["checksum"] != source_hash:
        raise ValueError(f"EDF does not match pinned Kuroda manifest: {path}")
    signal, fs, labels = read_edf(path)
    if fs != 1000 or len(signal) < fs * seconds:
        raise ValueError(f"Expected at least {seconds}s at 1 kHz: {path}")
    chosen = np.linspace(0, len(labels) - 1, min(max_channels, len(labels)), dtype=int)
    signal = signal[:fs * seconds, chosen]
    labels = [labels[int(i)] for i in chosen]
    rms_function, janca_function = load_epycom()
    hfo_a = detect_hfo(signal, fs, 0.0, labels)
    spike_a = detect_spike(signal, fs, 0.0, labels)
    hfo_b = []
    hfo_effective = []
    spike_b = []
    hfo_sos = butter(4, (80, 250), btype="bandpass", fs=fs, output="sos")
    for column, label in enumerate(labels):
        raw = signal[:, column].astype(np.float64)
        if not np.isfinite(raw).all() or np.std(raw) == 0:
            continue
        filtered = sosfiltfilt(hfo_sos, raw)
        hfo_b.extend(epycom_hfo(filtered, fs, label, rms_function,
                                package_window_size=5, center_shift=0))
        # epycom compute_rms returns the *first* sample of a centered
        # convolution. Its nominal 5-sample window thus contains only three
        # signal samples; size 9 contains five. Recenter its result by 2 ms.
        hfo_effective.extend(epycom_hfo(filtered, fs, label, rms_function,
                                        package_window_size=9, center_shift=2))
        # The Janca comparator uses the same 10-60 Hz band but fits a different
        # log-normal envelope baseline. Its k1 is not a 6-MAD equivalent.
        spike_b.extend(convert_janca(janca_function(raw, fs=fs, bandwidth=(10, 60),
                                                    line_freq=60), label, fs))
    def normalize(events):
        return [{"label": e["label"], "start_s": float(e["start_s"]),
                 "peak_s": float(e["peak_s"]), "end_s": float(e["end_s"])} for e in events]
    hfo_a, spike_a = normalize(hfo_a), normalize(spike_a)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, events in (("hfo_current", hfo_a), ("hfo_epycom_nominal", hfo_b),
                         ("hfo_epycom_effective", hfo_effective),
                         ("spike_current", spike_a), ("spike_janca", spike_b)):
        with (OUT / f"sub-{subject}_{seconds}s_{max_channels}ch_{kind}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("label", "start_s", "peak_s", "end_s"))
            writer.writeheader()
            writer.writerows(events)
    return {"dataset": "OpenNeuro ds006107", "version": "1.0.0", "subject": subject,
            "source_sha256": source_hash, "sampling_hz": fs, "duration_s": seconds,
            "channels": labels, "annotation_files_used": False,
            "implementations": {"RealTimeHFO": revision(ROOT / "build" / "RealTimeHFO"),
                                "RealTimeSpike": revision(ROOT / "build" / "RealTimeSpike"),
                                "epycom": "0.3", "numpy": np.__version__, "scipy": scipy.__version__},
            "common_settings": {"montage": "original EDF channels, no rereference",
                                "hfo_band_hz": [80, 250], "hfo_rms_window_ms": 5,
                                "hfo_threshold_sd": 5, "hfo_merge_gap_ms": 10,
                                "hfo_duration_ms": [6, 200], "hfo_min_positive_peaks": 4,
                                "spike_band_hz": [10, 60], "spike_notch": "none",
                                "hfo_match_margin_ms": 10, "spike_match_margin_ms": 100},
            "nonidentical_settings": {"spike_current_threshold": "median envelope + 6 * 1.4826 * MAD",
                                      "spike_janca_threshold": "log-normal k1=3.65, k2=3.65",
                                      "hfo_rms_implementation": "SciPy uniform_filter1d versus epycom first sample of centered convolution"},
            "hfo_nominal_window": score_events(hfo_a, hfo_b, 0.01, labels),
            "hfo_effective_window": score_events(hfo_a, hfo_effective, 0.01, labels),
            "spike": score_events(spike_a, spike_b, 0.1, labels, point_mode=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+", help="Kuroda/OpenNeuro subject IDs")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--channels", type=int, default=8)
    args = parser.parse_args()
    reports = []
    for subject in args.subjects:
        report = evaluate(subject, args.seconds, args.channels)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    (OUT / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
