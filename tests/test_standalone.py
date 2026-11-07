"""Focused numerical checks for the standalone pipeline."""

import unittest
import io
import struct
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from offline.besa_reader import _decode_packed, channel_labels
from realtimemi.mi import modulation_index_gpu


class PackedDecoderTest(unittest.TestCase):
    def test_first_scheme_and_raw_escape(self):
        # First two values, two packed pairs, then one escaped int8.
        payload = b"\x0a\x00\x02\x00" + bytes([7 * 15 + 7, 8 * 15 + 6, 254, 0xFD])
        np.testing.assert_array_equal(_decode_packed(payload, 1, 7, False),
                                      [10, 2, 0, 0, 1, -1, -3])


class ChannelLabelTest(unittest.TestCase):
    def test_labels_follow_channel_index_not_tag_order(self):
        def tag(name, value):
            return struct.pack("<4sI", name, len(value)) + value

        header_payload = tag(b"OFBI", struct.pack("<Q", 64))
        header = tag(b"BCF1", header_payload)
        calibration = b"\x00" * 8 + tag(b"CHNR", struct.pack("<H", 3))
        for index, label in ((2, "B1"), (0, "DC01"), (1, "A1")):
            calibration += tag(b"CHLA", struct.pack("<H", index) + label.encode("utf-16le"))
        recording = io.BytesIO(header.ljust(64, b"\x00") + tag(b"BCAL", calibration))
        self.assertEqual(channel_labels(recording), ["DC01", "A1", "B1"])


class DemoDataTest(unittest.TestCase):
    def test_public_demo_is_five_minutes_and_varies_by_channel(self):
        path = Path(__file__).resolve().parents[1] / "demo" / "synthetic_100ch_5min_1khz.npz"
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(int(archive["sample_rate_hz"]), 1000)
            data = archive["data"]
        self.assertEqual(data.shape, (300000, 100))
        self.assertEqual(data.dtype, np.int16)
        self.assertFalse(np.array_equal(data[:, 0], data[:, 99]))


class GPUAlgorithmTest(unittest.TestCase):
    def test_gpu_matches_independent_fft_reference(self):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() == 0:
                self.skipTest("No CUDA GPU")
        except (ImportError, RuntimeError):
            self.skipTest("CuPy GPU runtime unavailable")

        rate = 1000
        seconds = 35
        t = np.arange(rate * seconds) / rate
        phase = 2 * np.pi * 3.5 * t
        coupled = np.sin(phase) + (1 + 0.8 * np.cos(phase)) * np.sin(2 * np.pi * 130 * t)
        control = np.sin(2 * np.pi * 3.5 * t) + np.sin(2 * np.pi * 130 * t)
        data = np.column_stack((coupled, control)).astype(np.float32)
        actual, windows = modulation_index_gpu(data, rate)
        self.assertEqual(actual.shape, (1, 2))
        np.testing.assert_array_equal(windows, [[0, 30]])

        delay = 1500
        lag = np.arange(-delay, delay + 1)
        taper = np.hanning(len(lag))

        def analytic(x, low, high):
            bandwidth = (high - low) / rate
            kernel = (2 * bandwidth * np.sinc(bandwidth * lag) * taper
                      * np.exp(1j * np.pi * (low + high) * lag / rate))
            return fftconvolve(np.pad(x.astype(float), delay, mode="reflect"),
                               kernel, mode="same")[delay:-delay]

        for channel in range(2):
            labels = np.minimum(np.floor((np.angle(analytic(data[:, channel], 3, 4))[:30000]
                                          + np.pi) * 18 / (2 * np.pi)).astype(int), 17)
            amp = np.abs(analytic(data[:, channel], 80, 250))[:30000]
            counts = np.bincount(labels, minlength=18)
            means = np.bincount(labels, weights=amp, minlength=18) / counts
            p = means / means.sum()
            expected = np.sum(p * np.log(18 * p)) / np.log(18)
            self.assertAlmostEqual(float(actual[0, channel]), float(expected), delta=1e-5)
        self.assertGreater(actual[0, 0], actual[0, 1])


if __name__ == "__main__":
    unittest.main()
