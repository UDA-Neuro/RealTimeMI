"""Compare real BESA GPU MI with an independent SciPy Tort calculation."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, firwin

from besa_reader import read_resampled_parallel


def reference(signal):
    rate = 1000
    delay = 1500
    lag = np.arange(-delay, delay + 1)
    taper = np.hanning(lag.size)
    padded = np.pad(signal.astype(np.float64), delay, mode="reflect")

    def analytic(low, high):
        bandwidth = (high - low) / rate
        kernel = (2 * bandwidth * np.sinc(bandwidth * lag) * taper
                  * np.exp(1j * np.pi * (low + high) / rate * lag))
        return fftconvolve(padded, kernel, mode="same")[delay:-delay]

    phase = np.angle(analytic(3, 4))
    amplitude = np.abs(analytic(80, 250))
    output = []
    for start in range(0, 271, 15):
        interval = slice(start * rate, (start + 30) * rate)
        ids = np.minimum(np.floor((phase[interval] + np.pi) * 18 / (2 * np.pi)).astype(int), 17)
        counts = np.bincount(ids, minlength=18)
        sums = np.bincount(ids, weights=amplitude[interval], minlength=18)
        if np.any(counts == 0) or not np.any(sums > 0):
            output.append(np.nan)
            continue
        p = sums / counts
        p /= p.sum()
        p = p[p > 0]
        output.append(np.sum(p * np.log(18 * p)) / np.log(18))
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gpu-csv", default="build/besa-mi-result.csv")
    parser.add_argument("--dll", default="build/besa_packed.dll")
    args = parser.parse_args()
    os.environ["BESA_PACKED_DLL"] = str(Path(args.dll).resolve())
    taps = firwin(301, 350, fs=10000, window=("kaiser", 8.6))
    data, rate = read_resampled_parallel(args.input, 0, 300, taps, workers=8)
    assert rate == 1000
    gpu = np.loadtxt(args.gpu_csv, delimiter=",")[:, 2:]
    # These channels have no ADC rail clipping in the initial data block.
    chosen = (4, 30, 70)
    for channel in chosen:
        expected = reference(data[:, channel])
        actual = gpu[:, channel]
        valid = np.isfinite(expected) & np.isfinite(actual)
        if not np.array_equal(np.isfinite(expected), np.isfinite(actual)):
            raise AssertionError(f"MI validity differs for channel {channel}")
        error = np.max(np.abs(expected[valid] - actual[valid])) if valid.any() else 0
        print(f"channel={channel} valid={valid.sum()} max_abs_difference={error:.9g}", flush=True)
        assert error < 1e-6


if __name__ == "__main__":
    main()
