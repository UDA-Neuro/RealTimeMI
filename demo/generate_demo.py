"""Create the deterministic, fictional 100-channel five-minute ECoG demo.

The two-second template repeats to keep the public compressed file small.
It is a software test signal, not a physiological recording.
"""

from pathlib import Path

import numpy as np


def main() -> None:
    rate = 1000
    seconds = 300
    channels = 100
    template_seconds = 2
    t = np.arange(rate * template_seconds, dtype=np.float64) / rate
    template = np.empty((len(t), channels), dtype=np.int16)
    for channel in range(channels):
        shift = 2 * np.pi * channel / channels
        phase = 2 * np.pi * 3.5 * t + shift
        modulation = 1.0 + (0.05 + 0.9 * channel / (channels - 1)) * np.cos(phase)
        carrier = 75 * modulation * np.sin(2 * np.pi * 120 * t + shift)
        slow = 25 * np.sin(phase)
        background = 8 * np.sin(2 * np.pi * 177 * t + shift / 3)
        template[:, channel] = np.rint(carrier + slow + background).astype(np.int16)
    data = np.tile(template, (seconds // template_seconds, 1))
    output = Path(__file__).with_name("synthetic_100ch_5min_1khz.npz")
    np.savez_compressed(output, data=np.asfortranarray(data), sample_rate_hz=np.int64(rate))
    print(f"{output}: {data.shape}, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
