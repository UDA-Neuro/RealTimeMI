"""Pilot numerical and injected-coupling checks for RealTimeMI."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from read_public_ieeg import read_edf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "validation" / "mi"
sys.path.insert(0, str(ROOT))
from realtimemi.mi import modulation_index_gpu  # noqa: E402


def cpu_reference(signal: np.ndarray, fs: int = 1000, window_s: int = 30) -> float:
    """Independent SciPy convolution and normalized KL for the first window."""
    delay = round(1.5 * fs)
    lag = np.arange(-delay, delay + 1)
    taper = 0.5 + 0.5 * np.cos(np.pi * lag / delay)
    padded = np.pad(signal, delay, mode="reflect")
    def analytic(low, high):
        bandwidth = (high - low) / fs
        kernel = (2 * bandwidth * np.sinc(bandwidth * lag) * taper *
                  np.exp(1j * np.pi * (low + high) * lag / fs))
        return fftconvolve(padded, kernel, mode="full")[2 * delay:2 * delay + len(signal)]
    phase = np.angle(analytic(3, 4))
    amplitude = np.abs(analytic(80, 250))
    ids = np.minimum(np.floor((phase[:window_s * fs] + np.pi) * 18 / (2 * np.pi)).astype(int), 17)
    counts = np.bincount(ids, minlength=18)
    means = np.bincount(ids, weights=amplitude[:window_s * fs], minlength=18) / counts
    p = means / means.sum()
    return float(np.sum(p * np.log(18 * p)) / np.log(18))


def main():
    (OUT / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(OUT / "tmp")
    fs = 1000
    t = np.arange(90 * fs) / fs
    phase = 2 * np.pi * 3.5 * t
    strengths = [0, 0.25, 0.5, 0.75]
    rng = np.random.default_rng(20261108)
    data = np.column_stack([
        np.sin(phase) + (1 + strength * np.cos(phase)) * np.sin(2 * np.pi * 130 * t)
        + rng.normal(0, 0.15, len(t)) for strength in strengths
    ]).astype(np.float32)
    started = time.perf_counter()
    mi30, windows30 = modulation_index_gpu(data, fs, window_seconds=30)
    mi60, windows60 = modulation_index_gpu(data, fs, window_seconds=60)
    reference = cpu_reference(data[:, -1], fs)
    report = {
        "software": "RealTimeMI", "seed": 20261108, "sample_rate_hz": fs,
        "duration_s": 90, "coupling_strengths": strengths,
        "mi30_means": np.mean(mi30, axis=0).tolist(),
        "mi60_means": np.mean(mi60, axis=0).tolist(),
        "windows30": windows30.tolist(), "windows60": windows60.tolist(),
        "cpu_reference_first_30s_strongest": reference,
        "gpu_first_30s_strongest": float(mi30[0, -1]),
        "absolute_error": abs(reference - float(mi30[0, -1])),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    public_file = (ROOT / "build" / "validation" / "data" / "nm000360" /
                   "sub-01" / "ieeg" / "sub-01_task-sleep_ieeg.edf")
    if public_file.is_file():
        public, rate, labels = read_edf(public_file)
        selected = [labels.index(name) for name in ("LA1", "LA2", "RA1", "RMH1")]
        signal = public[:90 * rate, selected]
        print("Public iEEG loaded; running GPU cross-check", flush=True)
        public_gpu, public_windows = modulation_index_gpu(signal, rate)
        print("GPU cross-check complete; running CPU reference", flush=True)
        public_cpu = [cpu_reference(signal[:, i], rate) for i in range(len(selected))]
        report["public_data"] = {
            "dataset": "nm000360", "subject": "01", "channels": [labels[i] for i in selected],
            "windows": len(public_windows), "gpu_first_window": public_gpu[0].tolist(),
            "cpu_first_window": public_cpu,
            "max_absolute_error_first_window": float(np.max(np.abs(public_gpu[0] - public_cpu))),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
