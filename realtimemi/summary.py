"""Recording-wide MI summaries and cross-electrode z-scores."""

from __future__ import annotations

import re

import numpy as np


def summarize_mi(mi: np.ndarray, labels: list[str], *, include_dc: bool = False) -> dict:
    """Average valid MI windows per channel, then standardize across electrodes.

    The reference uses population SD (ddof=0). DC-prefixed auxiliary channels
    are excluded by default. Undefined means never enter the reference group.
    """
    mi = np.asarray(mi, dtype=float)
    if mi.ndim != 2 or mi.shape[1] != len(labels) or mi.shape[0] == 0:
        raise ValueError("MI must be a nonempty windows-by-channels matrix matching labels")
    valid = np.isfinite(mi)
    counts = valid.sum(axis=0)
    means = np.full(len(labels), np.nan)
    np.divide(np.where(valid, mi, 0).sum(axis=0), counts, out=means, where=counts > 0)
    electrode = np.array([include_dc or re.fullmatch(r"DC\d+", label, re.I) is None
                          for label in labels], dtype=bool)
    # A recording consisting only of DC-labelled channels still has a useful map.
    if not electrode.any():
        electrode[:] = True
    reference = electrode & np.isfinite(means)
    z = np.full(len(labels), np.nan)
    center = spread = None
    if reference.sum() >= 2:
        center = float(np.mean(means[reference]))
        spread = float(np.std(means[reference], ddof=0))
        if spread > 0:
            z[reference] = (means[reference] - center) / spread
    return {
        "mean_mi": means,
        "valid_windows": counts,
        "z_score": z,
        "included": electrode,
        "reference_mean": center,
        "reference_sd": spread,
        "reference_channels": int(reference.sum()),
        "ddof": 0,
    }
