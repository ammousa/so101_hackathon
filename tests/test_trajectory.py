from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from so101_hackathon.deploy.trajectory import CSVJointTrajectory, HardwareTrajectoryLeader
from so101_hackathon.deploy.runtime import hardware_obs_to_joint_positions
from scripts.deploy.deploy_traj import _return_to_start


class _FakeFollower:
    def __init__(self):
        """Initialize fake follower."""
        self.sent_actions = []

    def send_action(self, action):
        """Record sent action."""
        self.sent_actions.append(dict(action))


class TrajectoryTests(unittest.TestCase):
    def test_csv_trajectory_returns_one_row_per_step(self):
        """Verify trajectory playback returns absolute joint targets in order."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "targets.csv"
            csv_path.write_text(
                "time_s,shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper\n"
                "0.0,0.1,0.2,0.3,0.4,0.5,0.6\n"
                "0.0166667,1.1,1.2,1.3,1.4,1.5,1.6\n",
                encoding="utf-8",
            )
            trajectory = CSVJointTrajectory(csv_path=str(csv_path))

            first = trajectory.next_joint_target()
            second = trajectory.next_joint_target()

        self.assertEqual(first, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(second, [1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

    def test_csv_trajectory_repeats_configured_cycles_then_stops(self):
        """Verify trajectory playback repeats N times before stopping."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "targets.csv"
            csv_path.write_text(
                "0.1,0.2,0.3,0.4,0.5,0.6\n"
                "1.1,1.2,1.3,1.4,1.5,1.6\n",
                encoding="utf-8",
            )
            trajectory = CSVJointTrajectory(csv_path=str(csv_path), cycles=2)

            actions = [trajectory.next_joint_target() for _ in range(4)]

        self.assertEqual(
            actions,
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            ],
        )
        self.assertTrue(trajectory.completed)
        self.assertEqual(trajectory.start_target, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with self.assertRaises(StopIteration):
            trajectory.next_joint_target()

    def test_csv_trajectory_validates_return_to_start_steps(self):
        """Verify return-to-start step count must be non-negative."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "targets.csv"
            csv_path.write_text("0.1,0.2,0.3,0.4,0.5,0.6\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                CSVJointTrajectory(
                    csv_path=str(csv_path),
                    return_to_start_steps=-1,
                )

    def test_hardware_trajectory_leader_emits_hardware_fields(self):
        """Verify hardware trajectory leader matches deploy leader observation format."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "targets.csv"
            target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            csv_path.write_text(",".join(str(value) for value in target), encoding="utf-8")
            leader = HardwareTrajectoryLeader(CSVJointTrajectory(csv_path=str(csv_path)))

            action = leader.get_action()

        self.assertEqual(len(action), 6)
        for actual, expected in zip(hardware_obs_to_joint_positions(action), target):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_return_to_start_uses_saved_start_pose(self):
        """Verify cleanup returns to saved start pose, not first CSV row."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "targets.csv"
            csv_path.write_text("0.1,0.2,0.3,0.4,0.5,0.6\n", encoding="utf-8")
            trajectory = CSVJointTrajectory(
                csv_path=str(csv_path),
                return_to_start_steps=2,
            )
            follower = _FakeFollower()
            saved_start = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

            steps = _return_to_start(
                follower=follower,
                trajectory=trajectory,
                start_joint_pos=saved_start,
                lower_limits=[-10.0] * 6,
                upper_limits=[10.0] * 6,
                sleep_fn=lambda _seconds: None,
                fps=60,
                active_follower_joint_names=None,
            )

        self.assertEqual(steps, 2)
        self.assertEqual(len(follower.sent_actions), 2)
        returned = hardware_obs_to_joint_positions(follower.sent_actions[0])
        for actual, expected in zip(returned, saved_start):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
