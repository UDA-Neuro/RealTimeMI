"""Repeat the complete standalone path and optionally compare a MATLAB CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtimemi.pipeline import analyze_besa


def read_csv(path: Path, *, header: bool) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return np.array([[float(x) for x in row] for row in rows[int(header):]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="build/python-benchmark")
    parser.add_argument("--reference", help="Optional MATLAB MI CSV, without header")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [analyze_besa(args.input, output_dir) for _ in range(args.repeats)]
    observed_runs = [read_csv(Path(run["csv_path"]), header=True) for run in runs]
    report = {
        "software": "RealTimeMI Python standalone",
        "timings_s": [run["elapsed_s"]["result_ready"] for run in runs],
        "all_under_30_s": all(run["elapsed_s"]["result_ready"] < 30 for run in runs),
        "channels": runs[0]["source_channels"],
        "windows": runs[0]["windows"],
        "invalid_values": [run["invalid_values"] for run in runs],
        "repeat_numeric_identical": all(np.array_equal(observed_runs[0], other, equal_nan=True)
                                        for other in observed_runs[1:]),
        "gpu": runs[0]["gpu"],
    }
    if args.reference:
        reference = read_csv(Path(args.reference), header=False)
        observed = observed_runs[-1]
        if reference.shape != observed.shape:
            raise ValueError(f"Reference shape {reference.shape} != result shape {observed.shape}")
        report["reference_finite_mask_equal"] = bool(np.array_equal(
            np.isfinite(reference), np.isfinite(observed)))
        difference = np.abs(reference[:, 2:] - observed[:, 2:])
        report["reference_median_abs_delta"] = float(np.nanmedian(difference))
        report["reference_p95_abs_delta"] = float(np.nanquantile(difference, .95))
        report["reference_max_abs_delta"] = float(np.nanmax(difference))
    target = output_dir / "benchmark.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
