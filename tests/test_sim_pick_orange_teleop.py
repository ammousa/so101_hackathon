from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import scripts.deploy.sim_pick_orange.teleop as teleop_script
from so101_hackathon.deploy.runtime import (
    FixedDisturbanceChannel,
    build_follower_action,
    hardware_obs_to_joint_positions,
)
from so101_hackathon.utils.rl_utils import TELEOP_RESIDUAL_ACTION_SCALE


class _AbsoluteController:
    pass


class _ResidualController:
    action_mode = "residual"


class SimPickOrangeTeleopTests(unittest.TestCase):
    def test_parser_help_exposes_controller_options(self):
        """Verify parser help exposes controller options."""
        parser = teleop_script.build_parser()

        help_text = parser.format_help()
        args = parser.parse_args(
            ["--controller", "pd", "--controller-coeff", "0.25"])

        self.assertIn("--controller", help_text)
        self.assertIn("--controller-config", help_text)
        self.assertIn("--checkpoint-path", help_text)
        self.assertIn("--disturbance-channel", help_text)
        self.assertEqual(args.controller, "pd")
        self.assertEqual(args.controller_coeff, 0.25)
        self.assertEqual(parser.parse_args([]).disturbance_channel, "fixed")

    def test_ultrazohm_rejects_multi_env_pick_orange(self):
        """Verify UltraZohm disturbance rejects multi-env PickOrange teleop."""
        args = teleop_script.build_parser().parse_args(
            ["--disturbance-channel", "ultrazohm", "--num_envs", "2"])

        with self.assertRaisesRegex(ValueError, "--num_envs 1"):
            teleop_script.validate_disturbance_args(args)

    def test_action_disturbance_accepts_unbatched_action_vector(self):
        """Verify action disturbance accepts an unbatched action vector."""
        channel = FixedDisturbanceChannel(delay_steps=0, noise_std=0.0, seed=0)

        disturbed = teleop_script.apply_action_disturbance(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            channel,
        )

        self.assertEqual(disturbed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_ultrazohm_disturbance_accepts_unbatched_action_vector(self):
        """Verify UltraZohm disturbance accepts an unbatched action vector."""
        channel = mock.Mock()
        channel.apply.return_value = build_follower_action(
            [1.1, 2.1, 3.1, 4.1, 5.1, 0.7])

        disturbed = teleop_script.apply_ultrazohm_action_disturbance(
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.6],
            channel,
        )

        expected = hardware_obs_to_joint_positions(channel.apply.return_value)
        for actual, expected_value in zip(disturbed, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)

    def test_ultrazohm_disturbance_errors_can_be_fallback_safe(self):
        """Verify UltraZohm disturbance errors are ordinary exceptions."""
        channel = mock.Mock()
        channel.apply.side_effect = OSError("CAN interface hiccup")

        with self.assertRaisesRegex(OSError, "CAN interface hiccup"):
            teleop_script.apply_ultrazohm_action_disturbance(
                [1.0, 2.0, 3.0, 4.0, 5.0, 0.6],
                channel,
            )

    def test_build_controller_config_forwards_runtime_defaults(self):
        """Verify build controller config forwards runtime defaults."""
        args = argparse.Namespace(
            controller_config=None,
            device="cpu",
            seed=None,
            checkpoint_path="/tmp/model.pt",
        )

        config = teleop_script.build_controller_config(args, seed=123)

        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["seed"], 123)
        self.assertEqual(config["checkpoint_path"], "/tmp/model.pt")

    def test_build_controller_config_preserves_yaml_overrides(self):
        """Verify build controller config preserves yaml overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "controller.yaml"
            config_path.write_text(
                "kp: 2.0\ndevice: cuda:0\n", encoding="utf-8")
            args = argparse.Namespace(
                controller_config=str(config_path),
                device="cpu",
                seed=42,
                checkpoint_path=None,
            )

            config = teleop_script.build_controller_config(args, seed=123)

        self.assertEqual(config["kp"], 2.0)
        self.assertEqual(config["device"], "cuda:0")
        self.assertEqual(config["seed"], 123)

    def test_absolute_controller_action_blends_toward_controller_output(self):
        """Verify absolute controller action blends toward controller output."""
        action = teleop_script.adapt_controller_action(
            leader_action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            controller_action=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            controller=_AbsoluteController(),
            controller_coeff=0.25,
        )

        self.assertEqual(action, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

    def test_residual_controller_action_adds_clamped_residual(self):
        """Verify residual controller action adds clamped residual."""
        action = teleop_script.adapt_controller_action(
            leader_action=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            controller_action=[2.0, -2.0, 0.5, 0.0, 1.0, -1.0],
            controller=_ResidualController(),
            controller_coeff=0.5,
        )

        self.assertEqual(
            action,
            [
                1.0 + 0.5 * TELEOP_RESIDUAL_ACTION_SCALE,
                1.0 - 0.5 * TELEOP_RESIDUAL_ACTION_SCALE,
                1.0 + 0.25 * TELEOP_RESIDUAL_ACTION_SCALE,
                1.0,
                1.0 + 0.5 * TELEOP_RESIDUAL_ACTION_SCALE,
                1.0 - 0.5 * TELEOP_RESIDUAL_ACTION_SCALE,
            ],
        )

    def test_sim_observation_builder_produces_flat_teleop_layout(self):
        """Verify sim observation builder produces flat teleop layout."""
        builder = teleop_script.SimTeleopObservationBuilder()

        obs = builder.build(
            leader_joint_pos=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            follower_joint_pos=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            dt=0.5,
        )
        builder.set_previous_action([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        next_obs = builder.build(
            leader_joint_pos=[101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            follower_joint_pos=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            dt=0.5,
        )
        negative_obs = builder.build(
            leader_joint_pos=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            follower_joint_pos=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            dt=0.5,
        )

        self.assertEqual(len(obs), 30)
        self.assertEqual(obs[6:12], [0.0] * 6)
        self.assertEqual(obs[12:18], [0.5] * 6)
        self.assertEqual(obs[24:30], [0.0] * 6)
        self.assertEqual(next_obs[6:12], [100.0] * 6)
        self.assertEqual(next_obs[24:30], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        self.assertEqual(negative_obs[6:12], [-100.0] * 6)

    def test_tensor_observation_and_residual_action_match_sim_shapes(self):
        """Verify tensor observation and residual action match sim shapes."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        builder = teleop_script.SimTeleopObservationBuilder()
        leader = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        follower = torch.zeros_like(leader)

        obs = builder.build(
            leader_joint_pos=leader,
            follower_joint_pos=follower,
            dt=1.0 / 60.0,
        )
        next_obs = builder.build(
            leader_joint_pos=leader + 2.0,
            follower_joint_pos=follower,
            dt=1.0 / 60.0,
        )
        action = teleop_script.adapt_controller_action(
            leader_action=leader,
            controller_action=torch.tensor([2.0, -2.0, 0.5, 0.0, 1.0, -1.0]),
            controller=_ResidualController(),
            controller_coeff=0.5,
        )

        self.assertEqual(tuple(obs.shape), (1, 30))
        self.assertTrue(torch.allclose(next_obs[:, 6:12], torch.full_like(leader, 100.0)))
        self.assertTrue(torch.allclose(
            action,
            torch.tensor([[1.125, 0.875, 1.0625, 1.0, 1.125, 0.875]]),
        ))

    def test_keyboard_state_starts_immediately_when_app_window_is_missing(self):
        """Verify keyboard state starts immediately when app window is missing."""
        fake_carb = types.SimpleNamespace(
            input=types.SimpleNamespace(
                acquire_input_interface=mock.Mock(),
                KeyboardEventType=types.SimpleNamespace(KEY_PRESS="KEY_PRESS"),
            )
        )
        fake_omni = types.SimpleNamespace(
            appwindow=types.SimpleNamespace(
                get_default_app_window=mock.Mock(return_value=None)
            )
        )

        with mock.patch.dict(sys.modules, {"carb": fake_carb, "omni": fake_omni}):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                keyboard = teleop_script.KeyboardTeleopState()

        self.assertTrue(keyboard.started)
        self.assertFalse(keyboard.pop_reset_requested())
        self.assertFalse(keyboard.pop_success_requested())
        self.assertIn("starting teleoperation immediately", stdout.getvalue())
        fake_carb.input.acquire_input_interface.assert_not_called()


if __name__ == "__main__":
    unittest.main()
