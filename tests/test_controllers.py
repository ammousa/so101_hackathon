from __future__ import annotations

import unittest

from so101_hackathon.controllers.rl_ppo import PPOController
from so101_hackathon.controllers.rule_based_pd import TeleopPDController
from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.utils.obs_utils import TELEOP_HISTORY_LENGTH, TELEOP_JOINT_NAMES


def _build_observation(latest_error: list[float], latest_error_vel: list[float]) -> list[float]:
    joint_dim = len(TELEOP_JOINT_NAMES)
    term_size = joint_dim * TELEOP_HISTORY_LENGTH
    history = [0.0] * (term_size * 5)

    def write_latest(term_index: int, values: list[float]) -> None:
        start = term_index * term_size + term_size - joint_dim
        history[start : start + joint_dim] = values

    write_latest(2, latest_error)
    write_latest(3, latest_error_vel)
    return history


class ControllerTests(unittest.TestCase):
    def test_pd_controller_uses_joint_error_and_velocity_and_clamps(self):
        controller = TeleopPDController(kp=2.0, kd=0.5, max_action=0.5)
        observation = _build_observation(
            latest_error=[0.1, -0.1, 0.4, -0.4, 1.0],
            latest_error_vel=[0.2, -0.2, 0.0, 0.0, 0.0],
        )

        action = controller.act(observation)

        self.assertEqual(len(action), 5)
        self.assertAlmostEqual(action[0], 0.3, places=6)
        self.assertAlmostEqual(action[1], -0.3, places=6)
        self.assertAlmostEqual(action[2], 0.5, places=6)
        self.assertAlmostEqual(action[3], -0.5, places=6)
        self.assertAlmostEqual(action[4], 0.5, places=6)

    def test_registry_lists_and_builds_pd_controller(self):
        self.assertEqual(list_controller_names(), ["pd", "ppo"])
        controller = create_controller("pd", env=None, config={"kp": 0.25})
        self.assertIsInstance(controller, TeleopPDController)
        self.assertEqual(controller.kp, 0.25)

    def test_registry_ignores_unrelated_kwargs_for_simple_controllers(self):
        controller = create_controller(
            "pd",
            env=None,
            config={"kp": 0.5, "device": "cuda:0", "seed": 7, "logger": "tensorboard"},
        )
        self.assertIsInstance(controller, TeleopPDController)
        self.assertEqual(controller.kp, 0.5)

    def test_ppo_controller_requires_environment_before_runtime_imports(self):
        with self.assertRaises(ValueError):
            PPOController(env=None)


if __name__ == "__main__":
    unittest.main()
