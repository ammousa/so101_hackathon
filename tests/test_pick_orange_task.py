from __future__ import annotations

import unittest

from so101_hackathon.deploy.runtime import FixedDisturbanceChannel
from scripts.deploy.sim_pick_orange.teleop import apply_action_disturbance, build_parser

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    torch = None


@unittest.skipIf(torch is None, "torch is required for PickOrange teleop tests")
class PickOrangeTeleopScriptTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.teleop_device, "so101leader")
        self.assertEqual(args.num_envs, 1)
        self.assertEqual(args.delay_steps, 0)
        self.assertEqual(args.noise_std, 0.0)

    def test_apply_action_disturbance_respects_delay_steps(self):
        channel = FixedDisturbanceChannel(delay_steps=1, noise_std=0.0, seed=7)
        actions_a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
        actions_b = torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=torch.float32)

        disturbed_a = apply_action_disturbance(actions_a, channel)
        disturbed_b = apply_action_disturbance(actions_b, channel)

        self.assertTrue(torch.equal(disturbed_a, actions_a))
        self.assertTrue(torch.equal(disturbed_b, actions_a))
