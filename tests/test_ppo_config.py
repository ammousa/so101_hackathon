from __future__ import annotations

import unittest

from so101_hackathon.rl_training.ppo_config import TeleopPpoRunnerCfg


class TeleopPpoRunnerCfgTests(unittest.TestCase):
    def test_to_dict_matches_current_rsl_rl_runner_schema(self):
        cfg = TeleopPpoRunnerCfg(empirical_normalization=True)

        runner_cfg = cfg.to_dict()

        self.assertIn("policy", runner_cfg)
        self.assertNotIn("actor", runner_cfg)
        self.assertNotIn("critic", runner_cfg)
        self.assertEqual(
            runner_cfg["obs_groups"],
            {
                "policy": ["policy"],
                "critic": ["policy"],
            },
        )
        self.assertEqual(
            runner_cfg["policy"],
            {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "noise_std_type": "scalar",
                "actor_obs_normalization": True,
                "critic_obs_normalization": True,
                "actor_hidden_dims": [256, 128, 64],
                "critic_hidden_dims": [256, 128, 64],
                "activation": "elu",
            },
        )
        self.assertEqual(runner_cfg["algorithm"]["class_name"], "PPO")


if __name__ == "__main__":
    unittest.main()
