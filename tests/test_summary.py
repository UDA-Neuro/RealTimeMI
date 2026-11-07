"""Cross-electrode z-score uses each channel's complete valid-window mean."""

import unittest

import numpy as np

from realtimemi.summary import summarize_mi


class SummaryTest(unittest.TestCase):
    def test_means_nan_and_dc_exclusion(self):
        mi = np.array([[100, 1, 3, np.nan], [100, 3, 5, np.nan]], float)
        result = summarize_mi(mi, ["DC01", "A1", "A2", "A3"])
        np.testing.assert_allclose(result["mean_mi"][:3], [100, 2, 4])
        np.testing.assert_allclose(result["z_score"][1:3], [-1, 1])
        self.assertTrue(np.isnan(result["z_score"][0]))
        self.assertTrue(np.isnan(result["z_score"][3]))
        self.assertEqual(result["reference_channels"], 2)
        self.assertEqual(result["valid_windows"].tolist(), [2, 2, 2, 0])

    def test_include_dc_and_zero_spread(self):
        result = summarize_mi(np.array([[1., 1.], [1., 1.]]), ["DC01", "A1"], include_dc=True)
        self.assertEqual(result["reference_channels"], 2)
        self.assertEqual(result["reference_sd"], 0)
        self.assertTrue(np.isnan(result["z_score"]).all())

    def test_generic_channels_are_included(self):
        result = summarize_mi(np.array([[1., 2., 3.]]), ["channel_001", "channel_002", "channel_003"])
        self.assertEqual(result["reference_channels"], 3)
        self.assertAlmostEqual(result["z_score"][1], 0)


if __name__ == "__main__":
    unittest.main()
