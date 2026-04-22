from __future__ import annotations

import math
import unittest

from so101_hackathon.utils.eval_metrics import TeleopMetricAccumulator


class TeleopMetricAccumulatorTests(unittest.TestCase):
    def test_summary_is_zero_before_any_episodes_finish(self):
        """Verify summary is zero before any episodes finish."""
        metrics = TeleopMetricAccumulator()

        summary = metrics.summary()

        self.assertEqual(summary["eval/joint_rmse"], 0.0)
        self.assertEqual(summary["eval/num_failures"], 0.0)
        self.assertEqual(summary["eval/num_episodes"], 0.0)

    def test_add_step_rejects_wrong_joint_count(self):
        """Verify add step rejects wrong joint count."""
        metrics = TeleopMetricAccumulator(joint_count=6)

        with self.assertRaises(ValueError):
            metrics.add_step(joint_error=[0.1] * 5, action_rate=0.0)

    def test_finish_episode_and_summary_average_metrics(self):
        """Verify finish episode and summary average metrics."""
        metrics = TeleopMetricAccumulator(failure_threshold=0.5)
        metrics.reset_episode()
        metrics.add_step(
            joint_error=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            action_rate=0.25,
            ee_position_error=0.2,
            ee_orientation_error=0.1,
        )
        episode_one = metrics.finish_episode()

        metrics.add_step(
            joint_error=[0.0] * 6,
            action_rate=0.5,
            ee_position_error=0.4,
            ee_orientation_error=0.2,
            failure=True,
        )
        episode_two = metrics.finish_episode()
        summary = metrics.summary()

        expected_rmse_one = math.sqrt(sum(value * value for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) / 6.0)
        self.assertAlmostEqual(episode_one["joint_rmse"], expected_rmse_one, places=6)
        self.assertEqual(episode_one["num_failures"], 1.0)
        self.assertEqual(episode_one["failure_rate"], 1.0)
        self.assertEqual(episode_two["num_failures"], 1.0)
        self.assertAlmostEqual(summary["eval/num_failures"], 2.0, places=6)
        self.assertAlmostEqual(summary["eval/command_smoothness"], 0.375, places=6)
        self.assertEqual(summary["eval/num_episodes"], 2.0)

