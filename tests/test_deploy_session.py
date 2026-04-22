from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest import mock

from so101_hackathon.controllers.raw import RawController
from so101_hackathon.controllers.rl_ppo import PPOController
from so101_hackathon.controllers.rule_based_pd import TeleopPDController
from so101_hackathon.deploy.metrics import DeployMetricAccumulator
from so101_hackathon.deploy.runtime import (
    LiveTeleopObservationBuilder,
    get_joint_limit_vectors,
    hardware_obs_to_joint_positions,
    parse_joint_limits_from_urdf,
)
from so101_hackathon.deploy.session import run_deploy_session
from so101_hackathon.sim.robots.so101_follower_spec import joint_radians_to_motor_value
from so101_hackathon.utils.rl_utils import TELEOP_RESIDUAL_ACTION_SCALE

import scripts.deploy.deploy as deploy_script


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _robot_obs(arm_deg: list[float]) -> dict[str, float]:
    """Handle robot obs."""
    return {f"{joint_name}.pos": float(arm_deg[index]) for index, joint_name in enumerate(JOINT_NAMES)}


class _FakeLeader:
    def __init__(self, actions: list[dict[str, float]]):
        """Initialize the object."""
        self.actions = [dict(action) for action in actions]
        self.index = 0
        self.connect_calls = 0
        self.disconnect_calls = 0

    def get_action(self):
        """Return action."""
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return dict(action)

    def connect(self):
        """Connect the device."""
        self.connect_calls += 1

    def disconnect(self):
        """Disconnect the device."""
        self.disconnect_calls += 1


class _FakeFollower:
    def __init__(self, observations: list[dict[str, float]]):
        """Initialize the object."""
        self.observations = [dict(observation) for observation in observations]
        self.index = 0
        self.sent_actions: list[dict[str, float]] = []
        self.connect_calls = 0
        self.disconnect_calls = 0

    def get_observation(self):
        """Return observation."""
        observation = self.observations[min(self.index, len(self.observations) - 1)]
        self.index += 1
        return dict(observation)

    def send_action(self, action):
        """Run send action."""
        self.sent_actions.append(dict(action))
        return dict(action)

    def connect(self):
        """Connect the device."""
        self.connect_calls += 1

    def disconnect(self):
        """Disconnect the device."""
        self.disconnect_calls += 1


class _RecordingRawController(RawController):
    def __init__(self):
        """Initialize the object."""
        self.observations: list[list[float]] = []

    def act(self, obs):
        """Record observations before returning raw actions."""
        self.observations.append(list(obs))
        return super().act(obs)


def _args(controller_coeff: float = 1.0) -> argparse.Namespace:
    """Handle args."""
    return argparse.Namespace(
        fps=60,
        teleop_time_s=None,
        print_every=0,
        controller_coeff=controller_coeff,
        delay_steps=0,
        noise_std=0.0,
        seed=0,
    )


class DeploySessionTests(unittest.TestCase):
    def test_run_deploy_session_with_raw_controller_tracks_leader(self):
        """Verify run deploy session with raw controller tracks leader."""
        leader = _FakeLeader([_robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])
        follower = _FakeFollower([_robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()

        steps = run_deploy_session(
            args=_args(),
            leader=leader,
            follower=follower,
            controller=RawController(),
            observation_builder=LiveTeleopObservationBuilder(),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            num_iterations=1,
        )

        self.assertEqual(steps, 1)
        self.assertEqual(len(follower.sent_actions), 1)
        self.assertAlmostEqual(follower.sent_actions[0]["shoulder_pan.pos"], 10.0, places=4)
        self.assertAlmostEqual(follower.sent_actions[0]["gripper.pos"], 60.0, places=4)
        self.assertEqual(metrics.summary()["num_steps"], 1.0)

    def test_run_deploy_session_with_pd_controller_returns_absolute_targets(self):
        """Verify run deploy session with pd controller returns absolute targets."""
        leader = _FakeLeader([_robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])
        follower = _FakeFollower([_robot_obs([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()

        steps = run_deploy_session(
            args=_args(),
            leader=leader,
            follower=follower,
            controller=TeleopPDController(kp=0.5, kd=0.0, max_action=1.0),
            observation_builder=LiveTeleopObservationBuilder(),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            num_iterations=1,
        )

        self.assertEqual(steps, 1)
        self.assertEqual(len(follower.sent_actions), 1)
        self.assertAlmostEqual(follower.sent_actions[0]["shoulder_pan.pos"], 5.0, places=4)
        self.assertAlmostEqual(follower.sent_actions[0]["gripper.pos"], 60.0, places=4)

    def test_run_deploy_session_with_env_free_ppo_controller_uses_mocked_policy(self):
        """Verify run deploy session with env free ppo controller uses mocked policy."""
        leader = _FakeLeader([_robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])
        follower = _FakeFollower([_robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()
        fake_policy = lambda _obs: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        with mock.patch("so101_hackathon.controllers.rl_ppo.load_env_free_ppo_policy", return_value=fake_policy):
            controller = PPOController(env=None, checkpoint_path="/tmp/model.pt")

        steps = run_deploy_session(
            args=_args(controller_coeff=1.0),
            leader=leader,
            follower=follower,
            controller=controller,
            observation_builder=LiveTeleopObservationBuilder(),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            num_iterations=1,
        )

        self.assertEqual(steps, 1)
        self.assertEqual(len(follower.sent_actions), 1)
        leader_joint_pos = hardware_obs_to_joint_positions(_robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        self.assertAlmostEqual(
            follower.sent_actions[0]["shoulder_pan.pos"],
            joint_radians_to_motor_value(
                "shoulder_pan", leader_joint_pos[0] + TELEOP_RESIDUAL_ACTION_SCALE * 0.1),
            places=4,
        )
        gripper_lower, gripper_upper = parse_joint_limits_from_urdf()["gripper"]
        expected_gripper_percent = 100.0 * (
            ((leader_joint_pos[-1] + TELEOP_RESIDUAL_ACTION_SCALE * 0.6) - gripper_lower)
            / (gripper_upper - gripper_lower)
        )
        self.assertAlmostEqual(follower.sent_actions[0]["gripper.pos"], expected_gripper_percent, places=4)

    def test_run_deploy_session_delay_steps_holds_previous_command(self):
        """Verify run deploy session delay steps holds previous command."""
        leader = _FakeLeader(
            [
                _robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                _robot_obs([20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
            ]
        )
        follower = _FakeFollower(
            [
                _robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                _robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ]
        )
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()
        args = _args()
        args.delay_steps = 1

        steps = run_deploy_session(
            args=args,
            leader=leader,
            follower=follower,
            controller=RawController(),
            observation_builder=LiveTeleopObservationBuilder(),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            num_iterations=2,
        )

        self.assertEqual(steps, 2)
        self.assertEqual(len(follower.sent_actions), 2)
        self.assertAlmostEqual(follower.sent_actions[0]["shoulder_pan.pos"], 10.0, places=4)
        self.assertAlmostEqual(follower.sent_actions[1]["shoulder_pan.pos"], 10.0, places=4)

    def test_run_deploy_session_previous_action_uses_controller_command_before_delay(self):
        """Verify previous action tracks controller intent before delay."""
        leader = _FakeLeader(
            [
                _robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                _robot_obs([20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
                _robot_obs([30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            ]
        )
        follower = _FakeFollower([_robot_obs([0.0] * 6)] * 3)
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()
        args = _args()
        args.delay_steps = 1
        controller = _RecordingRawController()

        run_deploy_session(
            args=args,
            leader=leader,
            follower=follower,
            controller=controller,
            observation_builder=LiveTeleopObservationBuilder(),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            num_iterations=3,
        )

        second_leader_command = hardware_obs_to_joint_positions(
            _robot_obs([20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))
        self.assertEqual(len(controller.observations), 3)
        self.assertAlmostEqual(
            controller.observations[2][-6],
            second_leader_command[0],
            places=6,
        )

    def test_run_deploy_session_can_skip_missing_follower_gripper(self):
        """Verify run deploy session can skip missing follower gripper."""
        leader = _FakeLeader([_robot_obs([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])
        follower_obs = _robot_obs([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        follower_obs.pop("gripper.pos")
        follower = _FakeFollower([follower_obs])
        metrics = DeployMetricAccumulator()
        lower_limits, upper_limits = get_joint_limit_vectors()

        steps = run_deploy_session(
            args=_args(),
            leader=leader,
            follower=follower,
            controller=RawController(),
            observation_builder=LiveTeleopObservationBuilder(missing_follower_joint_names={"gripper"}),
            metrics=metrics,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            sleep_fn=lambda _seconds: None,
            active_follower_joint_names=JOINT_NAMES[:-1],
            num_iterations=1,
        )

        self.assertEqual(steps, 1)
        self.assertNotIn("gripper.pos", follower.sent_actions[0])


class DeployScriptTests(unittest.TestCase):
    def test_deploy_main_disconnects_hardware_on_keyboard_interrupt(self):
        """Verify deploy main disconnects hardware on keyboard interrupt."""
        leader = _FakeLeader([_robot_obs([0.0] * 6)])
        follower = _FakeFollower([_robot_obs([0.0] * 6)])

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(deploy_script, "normalize_device_for_runtime", return_value=("cpu", False)):
                with mock.patch.object(deploy_script, "create_controller", return_value=RawController()):
                    with mock.patch.object(deploy_script, "load_leader_follower_hardware_dependencies", return_value=(object, object, object, object, lambda _seconds: None)):
                        with mock.patch.object(deploy_script, "create_leader_follower_pair", return_value=(leader, follower)):
                            with mock.patch.object(deploy_script, "run_deploy_session", side_effect=KeyboardInterrupt):
                                result = deploy_script.main(["--output-dir", tmpdir, "--controller", "raw"])

        self.assertEqual(result, 0)
        self.assertEqual(leader.connect_calls, 1)
        self.assertEqual(follower.connect_calls, 1)
        self.assertEqual(leader.disconnect_calls, 1)
        self.assertEqual(follower.disconnect_calls, 1)

    def test_deploy_main_returns_one_and_disconnects_on_hardware_error(self):
        """Verify deploy main returns one and disconnects on hardware error."""
        leader = _FakeLeader([_robot_obs([0.0] * 6)])
        follower = _FakeFollower([_robot_obs([0.0] * 6)])

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(deploy_script, "normalize_device_for_runtime", return_value=("cpu", False)):
                with mock.patch.object(deploy_script, "create_controller", return_value=RawController()):
                    with mock.patch.object(deploy_script, "load_leader_follower_hardware_dependencies", return_value=(object, object, object, object, lambda _seconds: None)):
                        with mock.patch.object(deploy_script, "create_leader_follower_pair", return_value=(leader, follower)):
                            with mock.patch.object(deploy_script, "run_deploy_session", side_effect=ConnectionError("There is no status packet!")):
                                result = deploy_script.main(["--output-dir", tmpdir, "--controller", "raw"])

        self.assertEqual(result, 1)
        self.assertEqual(leader.connect_calls, 1)
        self.assertEqual(follower.connect_calls, 1)
        self.assertEqual(leader.disconnect_calls, 1)
        self.assertEqual(follower.disconnect_calls, 1)
