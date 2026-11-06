"""Verify blockwise resampling against whole-segment resampling on 10 s."""
import argparse

import numpy as np
from scipy.signal import firwin, resample_poly

from besa_reader import read_resampled_parallel, read_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    taps = firwin(301, 350, fs=10000, window=("kaiser", 8.6))
    source, _ = read_segment(args.input, 0, 10)
    reference = resample_poly(source, 1, 10, axis=0, window=taps).astype(np.float32)
    actual, _ = read_resampled_parallel(args.input, 0, 10, taps, workers=2)
    difference = np.abs(reference - actual)
    print("max_difference", float(difference.max()), "mean_difference", float(difference.mean()))
    assert difference.max() < 0.001


if __name__ == "__main__":
    main()
