from __future__ import annotations

import unittest
from unittest import mock

from so101_hackathon.controllers.ppo_loader import load_env_free_ppo_policy


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return {
            "shape": tuple(shape),
            "dtype": dtype,
            "device": device,
        }


class _FakeTensorDict(dict):
    def __init__(self, payload, batch_size):
        super().__init__(payload)
        self.batch_size = batch_size


class _FakeMlpModel:
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg):
        self.obs = obs
        self.obs_groups = obs_groups
        self.obs_set = obs_set
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.obs_normalization = obs_normalization
        self.distribution_cfg = distribution_cfg
        self.loaded_state_dict = None
        self.strict = None
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = dict(state_dict)
        self.strict = strict

    def eval(self):
        self.eval_called = True
        return self


class PpoLoaderTests(unittest.TestCase):
    def test_load_env_free_ppo_policy_builds_actor_and_loads_actor_state_dict(self):
        payload = {"actor_state_dict": {"mlp.weight": 123}}

        with mock.patch(
            "so101_hackathon.controllers.ppo_loader._import_rsl_rl_inference_deps",
            return_value=(_FakeTorch, _FakeTensorDict, _FakeMlpModel),
        ):
            with mock.patch(
                "so101_hackathon.controllers.ppo_loader._load_checkpoint_payload",
                return_value=payload,
            ):
                policy = load_env_free_ppo_policy(
                    checkpoint_path="/tmp/model.pt",
                    device="cpu",
                    actor_hidden_dims=[256, 128, 64],
                    empirical_normalization=False,
                )

        self.assertEqual(policy.actor.device, "cpu")
        self.assertEqual(policy.actor.loaded_state_dict, {"mlp.weight": 123})
        self.assertTrue(policy.actor.eval_called)
        self.assertEqual(policy.actor.output_dim, 6)
        self.assertEqual(policy.actor.hidden_dims, [256, 128, 64])
        self.assertEqual(policy.actor.obs_groups, {"actor": ["policy"], "critic": ["policy"]})


if __name__ == "__main__":
    unittest.main()
