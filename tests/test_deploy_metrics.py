from __future__ import annotations

import unittest

from so101_hackathon.deploy.metrics import DeployMetricAccumulator


class DeployMetricAccumulatorTests(unittest.TestCase):
    def test_update_and_summary_accumulate_expected_values(self):
        metrics = DeployMetricAccumulator()

        metrics.update(
            step=0,
            timestamp_s=0.0,
            leader_joint_pos=[1.0] * 6,
            follower_joint_pos=[0.0] * 6,
            commanded_joint_pos=[0.5] * 6,
        )
        metrics.update(
            step=1,
            timestamp_s=0.1,
            leader_joint_pos=[0.5] * 6,
            follower_joint_pos=[0.0] * 6,
            commanded_joint_pos=[0.25] * 6,
        )

        summary = metrics.summary()
        payload = metrics.summary_payload()

        self.assertEqual(summary["num_steps"], 2.0)
        self.assertGreater(summary["joint_rmse"], 0.0)
        self.assertGreaterEqual(summary["max_joint_error"], 0.5)
        self.assertIn("summary", payload)
        self.assertIn("per_joint", payload)
        self.assertEqual(len(metrics.timeseries_rows()), 2)

    def test_formatters_return_readable_strings(self):
        metrics = DeployMetricAccumulator()
        metrics.update(
            step=0,
            timestamp_s=0.0,
            leader_joint_pos=[1.0] * 6,
            follower_joint_pos=[0.0] * 6,
            commanded_joint_pos=[0.5] * 6,
        )

        self.assertIn("RMSE=", metrics.format_status_line(iter_idx=1, hz=60.0))
        self.assertIn("shoulder_pan", metrics.format_last_joint_errors())
        self.assertIn("Final deploy metrics:", metrics.format_final_report())
