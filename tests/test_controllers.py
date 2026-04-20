from __future__ import annotations

import unittest
from unittest import mock

from so101_hackathon.controllers.raw import RawController
from so101_hackathon.controllers.rl_ppo import PPOController
from so101_hackathon.controllers.rule_based_pd import TeleopPDController
from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.utils.obs_utils import TELEOP_HISTORY_LENGTH, TELEOP_JOINT_NAMES


def _build_observation(
    latest_error: list[float],
    latest_error_vel: list[float],
    latest_joint_command: list[float] | None = None,
) -> list[float]:
    joint_dim = len(TELEOP_JOINT_NAMES)
    term_size = joint_dim * TELEOP_HISTORY_LENGTH
    history = [0.0] * (term_size * 5)

    def write_latest(term_index: int, values: list[float]) -> None:
        start = term_index * term_size + term_size - joint_dim
        history[start : start + joint_dim] = values

    write_latest(0, latest_joint_command or [0.0] * joint_dim)
    write_latest(2, latest_error)
    write_latest(3, latest_error_vel)
    return history


class ControllerTests(unittest.TestCase):
    def test_raw_controller_returns_latest_joint_command(self):
        controller = RawController()
        command = [0.2, -0.1, 0.4, -0.3, 0.05, 0.7]
        observation = _build_observation(
            latest_error=[0.0] * 6,
            latest_error_vel=[0.0] * 6,
            latest_joint_command=command,
        )

        action = controller.act(observation)

        self.assertEqual(action, command)

    def test_pd_controller_uses_joint_error_and_velocity_and_clamps(self):
        controller = TeleopPDController(kp=2.0, kd=0.5, max_action=0.5)
        leader_command = [0.5, -0.4, 0.2, -0.1, 0.8, 0.3]
        observation = _build_observation(
            latest_error=[0.1, -0.1, 0.4, -0.4, 1.0, 0.0],
            latest_error_vel=[0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
            latest_joint_command=leader_command,
        )

        action = controller.act(observation)

        self.assertEqual(len(action), 6)
        self.assertAlmostEqual(action[0], 0.7, places=6)
        self.assertAlmostEqual(action[1], -0.6, places=6)
        self.assertAlmostEqual(action[2], 0.3, places=6)
        self.assertAlmostEqual(action[3], -0.2, places=6)
        self.assertAlmostEqual(action[4], 0.3, places=6)
        self.assertAlmostEqual(action[5], 0.3, places=6)

    def test_registry_lists_and_builds_pd_controller(self):
        self.assertEqual(list_controller_names(), ["pd", "ppo", "raw"])
        controller = create_controller("pd", env=None, config={"kp": 0.25})
        self.assertIsInstance(controller, TeleopPDController)
        self.assertEqual(controller.kp, 0.25)

    def test_registry_builds_raw_controller(self):
        controller = create_controller("raw", env=None)
        self.assertIsInstance(controller, RawController)

    def test_registry_ignores_unrelated_kwargs_for_simple_controllers(self):
        controller = create_controller(
            "pd",
            env=None,
            config={"kp": 0.5, "device": "cuda:0", "seed": 7, "logger": "tensorboard"},
        )
        self.assertIsInstance(controller, TeleopPDController)
        self.assertEqual(controller.kp, 0.5)

    def test_ppo_controller_can_load_without_environment(self):
        fake_policy = object()
        with mock.patch(
            "so101_hackathon.controllers.rl_ppo.load_env_free_ppo_policy",
            return_value=fake_policy,
        ):
            controller = PPOController(env=None, checkpoint_path="/tmp/model.pt")

        self.assertIs(controller._policy, fake_policy)
        self.assertEqual(controller.resolved_checkpoint_path, "/tmp/model.pt")


if __name__ == "__main__":
    unittest.main()
