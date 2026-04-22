from __future__ import annotations

import unittest

from so101_hackathon.rl_training.ppo_config import TeleopPpoRunnerCfg


class TeleopPpoRunnerCfgTests(unittest.TestCase):
    def test_to_dict_matches_current_rsl_rl_runner_schema(self):
        """Verify to dict matches current rsl rl runner schema."""
        cfg = TeleopPpoRunnerCfg(empirical_normalization=True)

        runner_cfg = cfg.to_dict()

        self.assertNotIn("policy", runner_cfg)
        self.assertIn("actor", runner_cfg)
        self.assertIn("critic", runner_cfg)
        self.assertEqual(
            runner_cfg["obs_groups"],
            {
                "actor": ["policy"],
                "critic": ["policy"],
            },
        )
        self.assertEqual(
            runner_cfg["actor"],
            {
                "class_name": "MLPModel",
                "hidden_dims": [256, 128, 64],
                "activation": "elu",
                "obs_normalization": True,
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            },
        )
        self.assertEqual(
            runner_cfg["critic"],
            {
                "class_name": "MLPModel",
                "hidden_dims": [256, 128, 64],
                "activation": "elu",
                "obs_normalization": True,
            },
        )
        self.assertEqual(runner_cfg["algorithm"]["class_name"], "PPO")


if __name__ == "__main__":
    unittest.main()
