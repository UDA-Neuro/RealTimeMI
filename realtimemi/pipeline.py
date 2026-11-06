"""Local BESA to GPU MI pipeline; no MATLAB or cloud connection required."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import scipy
from scipy.signal import firwin, freqz

from offline.besa_reader import metadata, read_resampled_parallel
from realtimemi import __version__
from realtimemi.mi import modulation_index_gpu

Progress = Callable[[str], None]


def _native_decoder() -> str | None:
    if os.environ.get("BESA_PACKED_DLL"):
        return os.environ["BESA_PACKED_DLL"]
    candidates = [Path(__file__).parent / "native" / "besa_packed.dll",
                  Path(__file__).resolve().parents[1] / "build" / "besa_packed.dll"]
    for path in candidates:
        if path.is_file():
            os.environ["BESA_PACKED_DLL"] = str(path.resolve())
            return str(path.resolve())
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_besa(
    source: str | Path,
    output_dir: str | Path,
    *,
    start_seconds: float = 0,
    duration_seconds: float = 300,
    workers: int = 8,
    hash_input: bool = False,
    progress: Progress | None = None,
) -> dict:
    """Analyze a complete BESA recording or a 1 kHz NPZ demo segment."""
    import cupy as cp

    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if not source.is_file() or source.suffix.lower() not in (".besa", ".npz"):
        raise ValueError("Select an existing .besa or .npz file")
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda message: None)
    started = time.perf_counter()
    if start_seconds < 0 or duration_seconds < 30:
        raise ValueError("Start must be nonnegative and duration at least 30 seconds")
    if source.suffix.lower() == ".besa":
        with source.open("rb") as stream:
            rate, total_samples, channels, _ = metadata(stream)
        if rate != 10_000:
            raise ValueError(f"This release supports 10 kHz BESA input; file is {rate:g} Hz")
        if round((start_seconds + duration_seconds) * rate) > total_samples:
            raise ValueError("Requested segment extends past the BESA recording")
        native = _native_decoder()
        if native is None:
            notify("Native BESA decoder unavailable; Python decoding may exceed 30 seconds")
        taps = firwin(301, 350, fs=10_000, window=("kaiser", 8.6))
        _, response = freqz(taps, worN=[250, 500], fs=10_000)
        notify("Reading and reducing BESA data to 1 kHz…")
        eeg, output_rate = read_resampled_parallel(source, start_seconds, duration_seconds,
                                                   taps, workers=workers)
        if output_rate != 1000 or eeg.shape[1] != channels:
            raise RuntimeError("Unexpected resampled EEG shape or rate")
        resampling = {"method": "scipy.signal.resample_poly", "factor": 10,
                      "fir_taps": 301, "cutoff_hz": 350, "kaiser_beta": 8.6,
                      "gain_250_db": float(20 * np.log10(abs(response[0]))),
                      "gain_500_db": float(20 * np.log10(abs(response[1])))}
        input_type = "BESA"
    else:
        notify("Loading 1 kHz NPZ recording…")
        with np.load(source, allow_pickle=False) as archive:
            rate = int(archive["sample_rate_hz"])
            if rate != 1000:
                raise ValueError("NPZ input must have sample_rate_hz=1000")
            full = archive["data"]
            if full.ndim != 2 or not np.issubdtype(full.dtype, np.number):
                raise ValueError("NPZ data must be samples by channels")
            first = round(start_seconds * rate)
            last = first + round(duration_seconds * rate)
            if last > len(full):
                raise ValueError("Requested segment extends past the NPZ recording")
            channels = full.shape[1]
            eeg = np.ascontiguousarray(full[first:last])
        resampling = None
        native = None
        input_type = "NPZ"
    decoded = time.perf_counter()
    notify("Calculating MI on the local GPU…")
    mi, windows = modulation_index_gpu(eeg, 1000)
    del eeg
    computed = time.perf_counter()
    notify("Saving results…")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    csv_path = output_dir / f"mi-{run_id}.csv"
    json_path = output_dir / f"mi-{run_id}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_start_s", "window_end_s"] +
                        [f"channel_{i + 1:03d}" for i in range(channels)])
        for window, row in zip(windows, mi):
            writer.writerow([f"{window[0]:g}", f"{window[1]:g}"] +
                            ["nan" if not np.isfinite(x) else f"{x:.17g}" for x in row])
    saved = time.perf_counter()
    gpu_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    info = {
        "software": "RealTimeMI", "version": __version__, "run_id": run_id,
        "source_bytes": source.stat().st_size,
        "source_sha256": _sha256(source) if hash_input else None,
        "source_sampling_hz": rate, "source_channels": channels,
        "input_type": input_type,
        "segment_start_s": start_seconds, "segment_duration_s": duration_seconds,
        "resampling": resampling,
        "mi": {"method": "Tort normalized KL", "phase_hz": [3, 4],
               "amplitude_hz": [80, 250], "window_s": 30, "overlap": 0.5,
               "phase_bins": 18, "analytic_fir_taps": 3001,
               "recording_edge": "reflect"},
        "windows": len(windows), "valid_values": int(np.isfinite(mi).sum()),
        "invalid_values": int(np.isnan(mi).sum()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "cupy": cp.__version__,
        "gpu": str(gpu_name),
        "native_decoder": native is not None,
        "elapsed_s": {"decode_and_resample": round(decoded - started, 3),
                      "gpu_mi": round(computed - decoded, 3),
                      "csv_save": round(saved - computed, 3),
                      "result_ready": round(saved - started, 3)},
    }
    json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    info["csv_path"] = str(csv_path)
    info["metadata_path"] = str(json_path)
    notify(f"Complete: {len(windows)} windows, {channels} channels")
    return info
