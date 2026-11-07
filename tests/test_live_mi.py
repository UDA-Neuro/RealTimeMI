"""Window scheduling and local audit log checks for automatic viewer MI."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realtimemi.live_mi import LiveMIPlanner, LiveMIResult, latest_window_start
from realtimemi.live_mi_store import LiveMIResultStore


class LiveMISchedulerTest(unittest.TestCase):
    def test_follow_provisional_is_finalized_when_context_arrives(self):
        self.assertIsNone(latest_window_start(29.9))
        self.assertEqual(latest_window_start(30), 0)
        self.assertEqual(latest_window_start(45), 15)
        planner = LiveMIPlanner()
        planner.observe(0, 30)
        self.assertEqual(planner.pop(), 0)
        provisional = LiveMIResult(0, 30, (4,), np.array([0.01]), False, 0.1)
        planner.complete(provisional)
        planner.observe(0, 30)
        self.assertIsNone(planner.pop())
        planner.observe(0, 35)
        self.assertEqual(planner.pop(), 0)
        planner.complete(LiveMIResult(0, 30, (4,), np.array([0.011]), True, 0.1))
        planner.observe(0, 45)
        self.assertEqual(planner.pop(), 15)

    def test_replay_uses_display_time(self):
        planner = LiveMIPlanner(replay=True)
        planner.observe(29.9, 100)
        self.assertIsNone(planner.pop())
        planner.observe(30, 100)
        self.assertEqual(planner.pop(), 0)

    def test_store_preserves_provisional_and_final_updates(self):
        scratch = Path(__file__).resolve().parents[1] / "build" / "test-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            store = LiveMIResultStore(directory, ["DC01", "A1"])
            store.append(LiveMIResult(0, 30, (1,), np.array([0.01]), False, 0.2))
            store.append(LiveMIResult(0, 30, (1,), np.array([0.011]), True, 0.2))
            with store.csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["status"] for row in rows], ["provisional", "final"])
            self.assertEqual([row["channel_label"] for row in rows], ["A1", "A1"])
            self.assertEqual(json.loads(store.json_path.read_text(encoding="utf-8"))["mi"]["hop_s"], 15)


if __name__ == "__main__":
    unittest.main()
