"""Benchmark the complete BESA -> 1 kHz -> GPU MI -> CSV path.

Example: python offline/benchmark_besa_pipeline.py --input PATH --repeats 3
The input is only read. Temporary EEG arrays stay in the ignored build folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import firwin, freqz, resample_poly

from besa_reader import metadata, read_resampled_parallel, read_segment, read_segment_parallel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="BESA block decoding processes")
    parser.add_argument("--fused", action="store_true", help="Decode and resample each block in parallel")
    parser.add_argument("--no-matlab", action="store_true", help="Measure BESA decode and resampling only")
    parser.add_argument("--service", action="store_true", help="Start MATLAB before timed runs and reuse it")
    parser.add_argument("--native", action="store_true", help="Use prebuilt build/besa_packed.dll")
    args = parser.parse_args()
    if args.native:
        dll = Path("build/besa_packed.dll").resolve()
        if not dll.is_file():
            parser.error(f"Native BESA decoder missing: {dll}")
        os.environ["BESA_PACKED_DLL"] = str(dll)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    build = Path("build")
    build.mkdir(exist_ok=True)
    raw_path = build / "besa-resampled-temp.f32"
    csv_path = build / "besa-mi-result.csv"
    log_path = build / "besa-pipeline-benchmark.json"

    # 250 Hz must remain in-band; attenuation is needed before the new 500 Hz Nyquist.
    taps = firwin(301, 350, fs=10000, window=("kaiser", 8.6)).astype(np.float64)
    freqs, response = freqz(taps, worN=[250, 500], fs=10000)
    gains_db = 20 * np.log10(np.maximum(abs(response), 1e-15))
    print(f"Anti-alias FIR: 301 taps, gain at 250/500 Hz = {gains_db[0]:.3f}/{gains_db[1]:.1f} dB", flush=True)

    runs = []
    manifest = dict(source_sha256=sha256_file(args.input), source_bytes=args.input.stat().st_size,
                    python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                    native_decoder_sha256=sha256_file(dll) if args.native else None,
                    filter_taps=301, filter_cutoff_hz=350,
                    gain_250_db=float(gains_db[0]), gain_500_db=float(gains_db[1]),
                    segment_start_s=args.start, segment_duration_s=args.duration)
    service = None
    service_log = None
    service_dir = build / "mi-service"
    if args.service and args.no_matlab:
        parser.error("--service and --no-matlab cannot be combined")
    try:
        if args.service:
            with args.input.open("rb") as source:
                _, _, service_channels, _ = metadata(source)
            service_dir.mkdir(exist_ok=True)
            for name in ("request.marker", "done.marker", "error.txt", "stop.marker", "ready.marker"):
                (service_dir / name).unlink(missing_ok=True)
            env = os.environ.copy()
            env.update(MI_INPUT_PATH=str(raw_path.resolve()), MI_OUTPUT_PATH=str(csv_path.resolve()),
                       MI_NCHANNELS=str(service_channels), MI_SERVICE_DIR=str(service_dir.resolve()),
                       MATLAB_PREFDIR=str((build / "matlab-prefs").resolve()))
            service_log = (service_dir / "matlab.log").open("w", encoding="utf-8")
            service = subprocess.Popen(["matlab", "-batch", "run('offline/mi_service.m')"],
                                       env=env, stdout=service_log, stderr=subprocess.STDOUT)
            wait_start = time.perf_counter()
            while not (service_dir / "ready.marker").exists():
                if service.poll() is not None:
                    raise RuntimeError(f"MATLAB service exited: {service_log.name}")
                if time.perf_counter() - wait_start > 240:
                    raise TimeoutError("MATLAB service did not become ready")
                time.sleep(0.1)
            print(f"MATLAB service ready after {time.perf_counter()-wait_start:.1f} s (before timed runs)", flush=True)
        for run in range(args.repeats):
            started = time.perf_counter()
            if args.fused:
                if args.workers < 2:
                    raise ValueError("--fused requires --workers >= 2")
                down, output_rate = read_resampled_parallel(args.input, args.start, args.duration, taps, workers=args.workers)
                decoded = resampled = time.perf_counter()
                if output_rate != 1000:
                    raise ValueError(f"Expected 1 kHz output, got {output_rate}")
                print(f"run {run + 1}: fused BESA decode + resample {resampled - started:.3f} s", flush=True)
            else:
                reader = read_segment if args.workers == 1 else read_segment_parallel
                if args.workers == 1:
                    eeg, rate = reader(args.input, args.start, args.duration)
                else:
                    eeg, rate = reader(args.input, args.start, args.duration, workers=args.workers)
                decoded = time.perf_counter()
                print(f"run {run + 1}: BESA decode {decoded - started:.3f} s", flush=True)
                if rate != 10000:
                    raise ValueError(f"Expected 10 kHz source, got {rate}")
                down = resample_poly(eeg, 1, 10, axis=0, window=taps).astype(np.float32)
                del eeg
                resampled = time.perf_counter()
                print(f"run {run + 1}: resample {resampled - decoded:.3f} s", flush=True)
            if not np.isfinite(down).all():
                raise ValueError("Non-finite values after resampling")
            down.tofile(raw_path)
            channels = down.shape[1]
            samples = down.shape[0]
            del down
            written = time.perf_counter()
            print(f"run {run + 1}: temp write {written - resampled:.3f} s", flush=True)
            if args.no_matlab:
                item = dict(run=run + 1, channels=channels, samples_1khz=samples,
                            decode_s=round(decoded - started, 3), resample_s=round(resampled - decoded, 3),
                            write_s=round(written - resampled, 3), total_s=round(written - started, 3))
                runs.append(item)
                log_path.write_text(json.dumps(dict(manifest, runs=runs), indent=2), encoding="utf-8")
                print(json.dumps(item), flush=True)
                continue
            env = os.environ.copy()
            env["MATLAB_PREFDIR"] = str((build / "matlab-prefs").resolve())
            env.update(MI_INPUT_PATH=str(raw_path.resolve()), MI_OUTPUT_PATH=str(csv_path.resolve()), MI_NCHANNELS=str(channels))
            if args.service:
                (service_dir / "done.marker").unlink(missing_ok=True)
                (service_dir / "error.txt").unlink(missing_ok=True)
                (service_dir / "request.marker").touch()
                wait_start = time.perf_counter()
                while not (service_dir / "done.marker").exists():
                    error_path = service_dir / "error.txt"
                    if error_path.exists():
                        raise RuntimeError(error_path.read_text(encoding="utf-8"))
                    if service.poll() is not None:
                        raise RuntimeError(f"MATLAB service exited: {service_log.name}")
                    if time.perf_counter() - wait_start > 180:
                        raise TimeoutError("MATLAB MI result timed out")
                    time.sleep(0.05)
                result = None
            else:
                result = subprocess.run(
                    ["matlab", "-batch", "run('offline/run_besa_mi.m')"],
                    env=env, text=True, capture_output=True,
                )
            finished = time.perf_counter()
            if result is not None and result.returncode:
                print(result.stdout[-3000:], result.stderr[-3000:], file=sys.stderr)
                raise RuntimeError(f"MATLAB failed with exit code {result.returncode}")
            values = np.loadtxt(csv_path, delimiter=",")
            if values.shape != (19, channels + 2):
                raise ValueError(f"Unexpected result shape {values.shape}")
            item = dict(run=run + 1, channels=channels, samples_1khz=samples,
                        decode_s=round(decoded - started, 3), resample_s=round(resampled - decoded, 3),
                        write_s=round(written - resampled, 3), matlab_and_save_s=round(finished - written, 3),
                        total_s=round(finished - started, 3), valid_mi=int(np.isfinite(values[:, 2:]).sum()))
            runs.append(item)
            log_path.write_text(json.dumps(dict(manifest, runs=runs), indent=2), encoding="utf-8")
            print(json.dumps(item), flush=True)
    finally:
        if service is not None:
            (service_dir / "stop.marker").touch()
            try:
                service.wait(timeout=20)
            except subprocess.TimeoutExpired:
                service.terminate()
                service.wait(timeout=10)
        if service_log is not None:
            service_log.close()
        raw_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
