"""Tort modulation index for 1 kHz recordings using a local NVIDIA GPU."""

from __future__ import annotations

import numpy as np


def _analytic_kernel(low: float, high: float, rate: int, lag: np.ndarray) -> np.ndarray:
    bandwidth = (high - low) / rate
    delay = (lag.size - 1) // 2
    taper = 0.5 + 0.5 * np.cos(np.pi * lag / delay)
    return (2 * bandwidth * np.sinc(bandwidth * lag) * taper
            * np.exp(1j * np.pi * (low + high) * lag / rate))


def modulation_index_gpu(
    data: np.ndarray,
    sample_rate: int = 1000,
    phase_band: tuple[float, float] = (3.0, 4.0),
    amplitude_band: tuple[float, float] = (80.0, 250.0),
    window_seconds: int = 30,
    overlap: float = 0.5,
    bins: int = 18,
    batch_size: int = 4,
    window_start_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MI[window, channel] and [start, end] seconds.

    The filter, padding, and binning match offline/recording_mi_gpu.m.
    A supported NVIDIA GPU and CuPy are required.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required for local GPU MI analysis") from exc

    data = np.asarray(data)
    if data.ndim != 2 or data.size == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("data must be a nonempty samples-by-channels numeric array")
    if not np.isfinite(data).all():
        raise ValueError("data contains non-finite samples")
    if sample_rate <= 2 * amplitude_band[1]:
        raise ValueError("sampling rate must exceed twice the amplitude upper frequency")
    if not (0 < phase_band[0] < phase_band[1] < amplitude_band[0] < amplitude_band[1]):
        raise ValueError("invalid phase or amplitude band")
    if not (0 <= overlap < 1) or bins < 2 or batch_size < 1:
        raise ValueError("invalid overlap, phase bins, or batch size")
    width = round(window_seconds * sample_rate)
    hop = round(width * (1 - overlap))
    if width < 2 or hop < 1 or data.shape[0] < width:
        raise ValueError("recording is shorter than one analysis window")

    if window_start_samples is None:
        starts = np.arange(0, data.shape[0] - width + 1, hop, dtype=np.int64)
    else:
        starts = np.asarray(window_start_samples, dtype=np.int64)
        if (starts.ndim != 1 or not len(starts) or np.any(starts < 0) or
                np.any(starts + width > data.shape[0])):
            raise ValueError("window start is outside the recording")
    windows = np.column_stack((starts, starts + width)) / sample_rate
    result = np.full((len(starts), data.shape[1]), np.nan, dtype=np.float64)
    delay = round(1.5 * sample_rate)
    lag = np.arange(-delay, delay + 1, dtype=np.float64)
    nfft = 1 << (data.shape[0] + 4 * delay - 1).bit_length()

    try:
        phase_filter = cp.fft.fft(cp.asarray(_analytic_kernel(*phase_band, sample_rate, lag)), n=nfft)
        amplitude_filter = cp.fft.fft(cp.asarray(_analytic_kernel(*amplitude_band, sample_rate, lag)), n=nfft)
        for first in range(0, data.shape[1], batch_size):
            last = min(first + batch_size, data.shape[1])
            padded = np.pad(data[:, first:last].astype(np.float64), ((delay, delay), (0, 0)), mode="reflect")
            spectrum = cp.fft.fft(cp.asarray(padded), n=nfft, axis=0)
            phase = cp.fft.ifft(spectrum * phase_filter[:, None], axis=0)
            labels = cp.asnumpy(cp.minimum(
                cp.floor((cp.angle(phase[2 * delay:2 * delay + len(data)]) + cp.pi)
                         * (bins / (2 * cp.pi))).astype(cp.int32), bins - 1))
            del phase
            amplitude = cp.fft.ifft(spectrum * amplitude_filter[:, None], axis=0)
            amps = cp.asnumpy(cp.abs(amplitude[2 * delay:2 * delay + len(data)]))
            del amplitude, spectrum
            for local_col, col in enumerate(range(first, last)):
                for row, start in enumerate(starts):
                    sl = slice(start, start + width)
                    ids = labels[sl, local_col]
                    counts = np.bincount(ids, minlength=bins)
                    if np.any(counts == 0):
                        continue
                    sums = np.bincount(ids, weights=amps[sl, local_col], minlength=bins)
                    if not np.any(sums > 0):
                        continue
                    means = sums / counts
                    p = means / means.sum()
                    p = p[p > 0]
                    result[row, col] = np.clip(np.sum(p * np.log(bins * p)) / np.log(bins), 0, 1)
        cp.cuda.Stream.null.synchronize()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f"NVIDIA GPU is unavailable: {exc}") from exc
    return result, windows
