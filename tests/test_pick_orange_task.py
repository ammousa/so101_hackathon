from __future__ import annotations

import unittest
from unittest import mock

from so101_hackathon.deploy.runtime import FixedDisturbanceChannel, build_follower_action
from scripts.deploy.sim_pick_orange.teleop import (
    apply_action_disturbance,
    apply_ultrazohm_action_disturbance,
    build_parser,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    torch = None


@unittest.skipIf(torch is None, "torch is required for PickOrange teleop tests")
class PickOrangeTeleopScriptTests(unittest.TestCase):
    def test_parser_defaults(self):
        """Verify parser defaults."""
        args = build_parser().parse_args([])
        self.assertEqual(args.teleop_device, "so101leader")
        self.assertEqual(args.num_envs, 1)
        self.assertEqual(args.delay_steps, 0)
        self.assertEqual(args.noise_std, 0.0)
        self.assertEqual(args.disturbance_channel, "fixed")
        self.assertEqual(args.uzohm_can_iface, "can0")
        self.assertEqual(args.uzohm_timeout_s, 1.0)

    def test_apply_action_disturbance_respects_delay_steps(self):
        """Verify apply action disturbance respects delay steps."""
        channel = FixedDisturbanceChannel(delay_steps=1, noise_std=0.0, seed=7)
        actions_a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
        actions_b = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=torch.float32)

        disturbed_a = apply_action_disturbance(actions_a, channel)
        disturbed_b = apply_action_disturbance(actions_b, channel)

        self.assertTrue(torch.equal(disturbed_a, actions_a))
        self.assertTrue(torch.equal(disturbed_b, actions_a))

    def test_apply_ultrazohm_action_disturbance_round_trips_lerobot_values(self):
        """Verify UltraZohm action disturbance round trips through LeRobot action values."""
        channel = mock.Mock()
        channel.apply.return_value = build_follower_action([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        actions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)

        disturbed = apply_ultrazohm_action_disturbance(actions, channel)

        sent_to_uzohm = channel.apply.call_args.args[0]
        expected_sent = build_follower_action(actions[0].tolist())
        self.assertEqual(set(sent_to_uzohm), set(expected_sent))
        self.assertTrue(torch.allclose(disturbed, torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])))
