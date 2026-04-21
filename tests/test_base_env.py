from __future__ import annotations

import types
import unittest
from unittest import mock

from so101_hackathon.envs.base_env import BaseHackathonEnvBuilder


class _DummyBuilder(BaseHackathonEnvBuilder):
    def __init__(self):
        self.cfg_calls = []

    def require_isaac_stack(self) -> None:
        return None

    def build_env_cfg(self, **kwargs):
        self.cfg_calls.append(dict(kwargs))
        return {"cfg": kwargs}


class BaseEnvBuilderTests(unittest.TestCase):
    def test_make_env_requires_video_dir_when_recording(self):
        builder = _DummyBuilder()
        fake_gym = types.SimpleNamespace(make=mock.Mock(return_value=mock.Mock()))

        with mock.patch.dict("sys.modules", {"gymnasium": fake_gym}):
            with self.assertRaises(ValueError):
                builder.make_env(env_id="teleop-env", env_cfg={}, record_video=True)

    def test_make_env_wraps_video_and_rl_layers(self):
        builder = _DummyBuilder()
        fake_env = mock.Mock(name="env")
        wrapped_video_env = mock.Mock(name="video_env")
        wrapped_rl_env = mock.Mock(name="rl_env")
        fake_gym = types.SimpleNamespace(
            make=mock.Mock(return_value=fake_env),
            wrappers=types.SimpleNamespace(
                RecordVideo=mock.Mock(return_value=wrapped_video_env)
            ),
        )
        fake_runtime_utils = types.SimpleNamespace(
            validate_rgb_rendering=mock.Mock(return_value=(True, "ok"))
        )
        fake_rl_wrapper = types.SimpleNamespace(
            RslRlVecEnvWrapper=mock.Mock(return_value=wrapped_rl_env)
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "gymnasium": fake_gym,
                "so101_hackathon.rl_training.runtime_utils": fake_runtime_utils,
                "so101_hackathon.rl_training.rsl_rl_wrapper": fake_rl_wrapper,
            },
        ):
            env = builder.make_env(
                env_id="teleop-env",
                env_cfg={"seed": 1},
                enable_cameras=True,
                record_video=True,
                video_dir="/tmp/videos",
                video_length=42,
                wrap_for_rl=True,
            )

        self.assertIs(env, wrapped_rl_env)
        fake_gym.make.assert_called_once_with(
            "teleop-env",
            cfg={"seed": 1},
            render_mode="rgb_array",
        )
        fake_runtime_utils.validate_rgb_rendering.assert_called_once_with(fake_env)
        fake_gym.wrappers.RecordVideo.assert_called_once()
        _, kwargs = fake_gym.wrappers.RecordVideo.call_args
        self.assertEqual(kwargs["video_folder"], "/tmp/videos")
        self.assertEqual(kwargs["video_length"], 42)
        fake_rl_wrapper.RslRlVecEnvWrapper.assert_called_once_with(wrapped_video_env)
