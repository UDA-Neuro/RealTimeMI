"""Waveform controls transform only the display copy."""

import unittest

import numpy as np

from realtimemi.display_processing import prepare_display


class DisplayProcessingTest(unittest.TestCase):
    def test_reference_labels_and_source_unchanged(self):
        source = np.array([[1., 3., 7.], [2., 5., 9.]])
        original = source.copy()
        average, labels = prepare_display(source, ["A1", "A2", "A3"],
                                          reference="Common average (shown)")
        np.testing.assert_allclose(average.mean(axis=1), 0, atol=1e-12)
        self.assertEqual(labels, ["A1", "A2", "A3"])
        bipolar, labels = prepare_display(source, ["A1", "A2", "A3"],
                                          reference="Adjacent bipolar")
        np.testing.assert_allclose(bipolar, [[-2, -4], [-3, -4]])
        self.assertEqual(labels, ["A1–A2", "A2–A3"])
        np.testing.assert_array_equal(source, original)

    def test_bandpass_and_notch_attenuate_expected_tones(self):
        rate = 1000
        t = np.arange(rate * 5) / rate
        source = (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t) +
                  np.sin(2 * np.pi * 200 * t))[:, None]
        filtered, _ = prepare_display(source, ["A1"],
                                      bandpass="0.5–70 Hz", notch="60 Hz")
        core = filtered[rate:-rate, 0]
        times = t[rate:-rate]
        def amplitude(hz):
            return abs(2 * np.mean(core * np.exp(-2j * np.pi * hz * times)))
        self.assertGreater(amplitude(10), 0.8)
        self.assertLess(amplitude(60), 0.1)
        self.assertLess(amplitude(200), 0.1)


if __name__ == "__main__":
    unittest.main()
