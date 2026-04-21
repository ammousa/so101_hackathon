from __future__ import annotations

import math
import unittest

from so101_hackathon.deploy.runtime import (
    DEFAULT_NOISE_JOINT_INDICES,
    FixedDisturbanceChannel,
    LiveTeleopObservationBuilder,
    blend_with_leader,
    build_follower_action,
    clamp_joint_positions,
    hardware_obs_to_joint_positions,
    parse_joint_limits_from_urdf,
)
from so101_hackathon.sim.robots.so101_follower_spec import (
    joint_radians_to_motor_value,
    motor_value_to_joint_radians,
)
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES


def _robot_obs(positions_deg: list[float]) -> dict[str, float]:
    return {
        f"{joint_name}.pos": float(positions_deg[index])
        for index, joint_name in enumerate(TELEOP_JOINT_NAMES)
    }


class DeployRuntimeTests(unittest.TestCase):
    def test_observation_builder_initializes_velocities_and_previous_action(self):
        builder = LiveTeleopObservationBuilder()
        joint_limits = parse_joint_limits_from_urdf()
        gripper_lower, gripper_upper = joint_limits["gripper"]
        expected_gripper = gripper_lower + \
            0.6 * (gripper_upper - gripper_lower)

        live_obs = builder.build(
            leader_observation=_robot_obs(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            follower_observation=_robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            dt=1.0 / 60.0,
        )

        self.assertEqual(len(live_obs.observation), 30)
        self.assertEqual(live_obs.leader_joint_vel, [0.0] * 6)
        self.assertEqual(live_obs.joint_error_vel, [0.0] * 6)
        self.assertEqual(live_obs.previous_action, [0.0] * 6)
        self.assertAlmostEqual(live_obs.leader_joint_pos[0], motor_value_to_joint_radians(
            "shoulder_pan", 10.0), places=6)
        self.assertAlmostEqual(live_obs.follower_joint_pos[0], motor_value_to_joint_radians(
            "shoulder_pan", 1.0), places=6)
        self.assertAlmostEqual(
            live_obs.joint_error[0],
            motor_value_to_joint_radians(
                "shoulder_pan", 10.0) - motor_value_to_joint_radians("shoulder_pan", 1.0),
            places=6,
        )
        self.assertAlmostEqual(
            live_obs.leader_joint_pos[-1], expected_gripper, places=6)
        self.assertEqual(live_obs.observation[-6:], [0.0] * 6)

    def test_observation_builder_propagates_previous_action_and_velocity(self):
        builder = LiveTeleopObservationBuilder()
        builder.build(
            leader_observation=_robot_obs(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            follower_observation=_robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            dt=0.5,
        )
        builder.set_previous_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        live_obs = builder.build(
            leader_observation=_robot_obs(
                [15.0, 25.0, 35.0, 45.0, 55.0, 65.0]),
            follower_observation=_robot_obs([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            dt=0.5,
        )

        self.assertEqual(live_obs.previous_action, [
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(
            live_obs.observation[-6:], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected_leader_delta = motor_value_to_joint_radians("shoulder_pan", 15.0) - motor_value_to_joint_radians(
            "shoulder_pan",
            10.0,
        )
        self.assertAlmostEqual(
            live_obs.leader_joint_vel[0], expected_leader_delta / 0.5, places=6)
        expected_error_delta = (
            motor_value_to_joint_radians(
                "shoulder_pan", 15.0) - motor_value_to_joint_radians("shoulder_pan", 2.0)
        ) - (
            motor_value_to_joint_radians(
                "shoulder_pan", 10.0) - motor_value_to_joint_radians("shoulder_pan", 1.0)
        )
        self.assertAlmostEqual(
            live_obs.joint_error_vel[0], expected_error_delta / 0.5, places=6)

    def test_blend_and_clamp_helpers(self):
        leader = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        controller = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        self.assertEqual(blend_with_leader(leader, controller, 0.0), leader)
        for actual, expected in zip(blend_with_leader(leader, controller, 1.0), controller):
            self.assertAlmostEqual(actual, expected, places=8)
        for actual, expected in zip(
            blend_with_leader(leader, controller, 0.25),
            [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        ):
            self.assertAlmostEqual(actual, expected, places=8)
        self.assertEqual(
            clamp_joint_positions(
                [2.0, -2.0, 0.3, 0.4, 0.5, 0.6],
                lower_limits=[-1.0] * 6,
                upper_limits=[1.0] * 6,
            ),
            [1.0, -1.0, 0.3, 0.4, 0.5, 0.6],
        )

    def test_build_follower_action_converts_all_joint_positions_to_degrees(self):
        joint_limits = parse_joint_limits_from_urdf()
        gripper_lower, gripper_upper = joint_limits["gripper"]
        mid_gripper = gripper_lower + 0.5 * (gripper_upper - gripper_lower)
        action = build_follower_action(
            [math.pi / 2, 0.0, -math.pi / 4, math.pi, -math.pi / 6, math.pi / 3])

        self.assertAlmostEqual(action["shoulder_pan.pos"], joint_radians_to_motor_value(
            "shoulder_pan", math.pi / 2), places=6)
        self.assertAlmostEqual(action["elbow_flex.pos"], joint_radians_to_motor_value(
            "elbow_flex", -math.pi / 4), places=6)
        self.assertAlmostEqual(action["wrist_flex.pos"], joint_radians_to_motor_value(
            "wrist_flex", math.pi), places=6)
        action = build_follower_action(
            [math.pi / 2, 0.0, -math.pi / 4, math.pi, -math.pi / 6, mid_gripper])
        self.assertAlmostEqual(action["gripper.pos"], 50.0, places=6)

    def test_build_follower_action_can_skip_missing_gripper_joint(self):
        action = build_follower_action(
            [math.pi / 2, 0.0, -math.pi / 4, math.pi, -math.pi / 6, math.pi / 3],
            active_joint_names=TELEOP_JOINT_NAMES[:-1],
        )

        self.assertIn("shoulder_pan.pos", action)
        self.assertNotIn("gripper.pos", action)

    def test_observation_builder_can_fill_missing_follower_gripper_from_leader(self):
        builder = LiveTeleopObservationBuilder(
            missing_follower_joint_names={"gripper"})
        follower_observation = _robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        follower_observation.pop("gripper.pos")
        joint_limits = parse_joint_limits_from_urdf()
        gripper_lower, gripper_upper = joint_limits["gripper"]
        expected_gripper = gripper_lower + \
            0.6 * (gripper_upper - gripper_lower)

        live_obs = builder.build(
            leader_observation=_robot_obs(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            follower_observation=follower_observation,
            dt=1.0 / 60.0,
        )

        self.assertAlmostEqual(
            live_obs.follower_joint_pos[-1], expected_gripper, places=6)
        self.assertAlmostEqual(live_obs.joint_error[-1], 0.0, places=6)

    def test_gripper_observation_and_action_round_trip_between_percent_and_radians(self):
        joint_limits = parse_joint_limits_from_urdf()
        gripper_lower, gripper_upper = joint_limits["gripper"]
        expected_gripper = gripper_lower + \
            0.25 * (gripper_upper - gripper_lower)
        observation = _robot_obs([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])

        joint_positions = hardware_obs_to_joint_positions(observation)
        self.assertAlmostEqual(joint_positions[-1], expected_gripper, places=6)

        action = build_follower_action(joint_positions)
        self.assertAlmostEqual(action["gripper.pos"], 25.0, places=6)

    def test_hardware_obs_to_joint_positions_still_raises_for_unexpected_missing_joint(self):
        follower_observation = _robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        follower_observation.pop("gripper.pos")

        with self.assertRaises(KeyError):
            hardware_obs_to_joint_positions(follower_observation)

    def test_disturbance_channel_applies_fixed_delay(self):
        channel = FixedDisturbanceChannel(delay_steps=2, noise_std=0.0, seed=0)

        outputs = [
            channel.apply([1.0, 0.0, 0.0, 0.0, 5.0, 6.0]),
            channel.apply([2.0, 0.0, 0.0, 0.0, 7.0, 8.0]),
            channel.apply([3.0, 0.0, 0.0, 0.0, 9.0, 10.0]),
            channel.apply([4.0, 0.0, 0.0, 0.0, 11.0, 12.0]),
        ]

        self.assertEqual(outputs[0], [1.0, 0.0, 0.0, 0.0, 5.0, 6.0])
        self.assertEqual(outputs[1], [1.0, 0.0, 0.0, 0.0, 5.0, 6.0])
        self.assertEqual(outputs[2], [1.0, 0.0, 0.0, 0.0, 5.0, 6.0])
        self.assertEqual(outputs[3], [2.0, 0.0, 0.0, 0.0, 7.0, 8.0])

    def test_disturbance_channel_noise_only_affects_first_four_joints(self):
        self.assertEqual(DEFAULT_NOISE_JOINT_INDICES, (0, 1, 2, 3))
        channel = FixedDisturbanceChannel(
            delay_steps=0, noise_std=0.05, seed=7)

        output = channel.apply([0.0, 0.0, 0.0, 0.0, 5.0, 6.0])

        self.assertNotEqual(output[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(output[4:], [5.0, 6.0])

    def test_disturbance_channel_noise_is_seeded_and_resettable(self):
        channel = FixedDisturbanceChannel(
            delay_steps=0, noise_std=0.05, seed=7)

        first = channel.apply([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        second = channel.apply([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertNotEqual(first, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(first[4:], [0.0, 0.0])
        self.assertEqual(second[4:], [0.0, 0.0])
        self.assertNotEqual(first, second)

        channel.reset()
        repeated = channel.apply([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for actual, expected in zip(repeated, first):
            self.assertAlmostEqual(actual, expected, places=10)


if __name__ == "__main__":
    unittest.main()
