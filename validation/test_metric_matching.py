"""The pilot metrics enforce one-to-one matches and stated exclusions."""

import unittest

from evaluate_hfo import score as hfo_score
from evaluate_spike import score as spike_score


class MatchingTest(unittest.TestCase):
    def test_spike_one_prediction_cannot_match_two_marks(self):
        reference = [{"side": "R", "time": 1.0}, {"side": "R", "time": 1.04}]
        predicted = [{"side": "R", "time": 1.01}, {"side": "L", "time": 1.0}]
        result = spike_score(reference, predicted)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))

    def test_hfo_ripple_match_precedes_fast_ripple_exclusion(self):
        reference = [{"pair": "A1-2", "start": 1.0, "end": 1.05}]
        ignored = [{"pair": "A1-2", "start": 1.04, "end": 1.1}]
        predicted = [
            {"pair": "A1-2", "start": 1.02, "peak": 1.03, "end": 1.04},
            {"pair": "A1-2", "start": 1.07, "peak": 1.08, "end": 1.09},
            {"pair": "B1-2", "start": 1.02, "peak": 1.03, "end": 1.04},
        ]
        result = hfo_score(reference, predicted, ignored)
        self.assertEqual((result["tp"], result["fp"], result["fn"],
                          result["predictions_ignored"]), (1, 1, 0, 1))


if __name__ == "__main__":
    unittest.main()
