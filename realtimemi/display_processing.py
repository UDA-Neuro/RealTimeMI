"""Display-only EEG references and filters for 1 kHz waveform chunks."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


SAMPLE_RATE = 1000
BANDPASS = {"Off": None, "0.5–70 Hz": (0.5, 70),
            "1–100 Hz": (1, 100), "3–250 Hz": (3, 250)}
NOTCH = {"Off": None, "50 Hz": 50, "60 Hz": 60}
REFERENCE = ("Original", "Common average (shown)", "Adjacent bipolar")
SENSITIVITY = {"Auto": 1.0, "0.25×": 0.25, "0.5×": 0.5,
               "1×": 1.0, "2×": 2.0, "4×": 4.0, "8×": 8.0}


def prepare_display(data: np.ndarray, labels: list[str], *,
                    bandpass: str = "Off", notch: str = "Off",
                    reference: str = "Original") -> tuple[np.ndarray, list[str]]:
    """Return processed display traces without modifying source data.

    Filters are zero-phase and run over the currently visible chunk. A short
    leading chunk is shown unfiltered until enough samples are available.
    """
    if bandpass not in BANDPASS or notch not in NOTCH or reference not in REFERENCE:
        raise ValueError("Unknown waveform display setting")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != len(labels):
        raise ValueError("Waveform columns must match labels")
    values = data.astype(np.float64, copy=True)
    for index in range(values.shape[1]):
        finite = np.isfinite(values[:, index])
        replacement = float(np.median(values[finite, index])) if finite.any() else 0.0
        values[~finite, index] = replacement
    if reference == "Common average (shown)":
        values -= values.mean(axis=1, keepdims=True)
    elif reference == "Adjacent bipolar":
        values = values[:, :-1] - values[:, 1:]
        labels = [f"{left}–{right}" for left, right in zip(labels[:-1], labels[1:])]
    if len(values) >= 64 and values.shape[1]:
        if bandpass != "Off":
            low, high = BANDPASS[bandpass]
            sos = butter(4, (low, high), btype="bandpass", fs=SAMPLE_RATE, output="sos")
            values = sosfiltfilt(sos, values, axis=0)
        if notch != "Off":
            b, a = iirnotch(NOTCH[notch], 30, fs=SAMPLE_RATE)
            values = filtfilt(b, a, values, axis=0)
    return values, list(labels)
