from __future__ import annotations

import argparse
import unittest
from unittest import mock

import numpy as np

from so101_hackathon.rl_training import runtime_utils


class _FakeRenderEnv:
    def __init__(self, frames, *, render_error: Exception | None = None):
        self._frames = list(frames)
        self._render_error = render_error
        self.render_calls = 0
        self.step_calls = 0
        self.reset_calls = 0
        self.action_space = mock.Mock()
        self.action_space.sample.return_value = [0.0] * 6

    def reset(self):
        self.reset_calls += 1
        return [0.0] * 6

    def render(self):
        self.render_calls += 1
        if self._render_error is not None:
            raise self._render_error
        if self._frames:
            return self._frames.pop(0)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def step(self, action):
        del action
        self.step_calls += 1
        return [0.0] * 6, 0.0, False, {}


class RuntimeUtilsTests(unittest.TestCase):
    def test_normalize_device_for_runtime_falls_back_to_cpu_and_disables_video(self):
        with mock.patch.object(runtime_utils, "cuda_is_healthy", return_value=(False, "bad driver")):
            device, video_enabled = runtime_utils.normalize_device_for_runtime(
                requested_device="cuda:0",
                wants_video=True,
            )

        self.assertEqual(device, "cpu")
        self.assertFalse(video_enabled)

    def test_normalize_device_for_runtime_keeps_cpu_request(self):
        with mock.patch.object(runtime_utils, "cuda_is_healthy") as mocked_health:
            device, video_enabled = runtime_utils.normalize_device_for_runtime(
                requested_device="cpu",
                wants_video=True,
            )

        self.assertEqual(device, "cpu")
        self.assertTrue(video_enabled)
        mocked_health.assert_not_called()

    def test_parse_version_tuple_stops_at_non_numeric_suffix(self):
        self.assertEqual(runtime_utils._parse_version_tuple("535.129.03"), (535, 129, 3))
        self.assertEqual(runtime_utils._parse_version_tuple("535.129.beta"), (535, 129))

    def test_apply_video_renderer_fallback_appends_flags_once(self):
        args = argparse.Namespace(video=True, kit_args="")

        with mock.patch.object(runtime_utils, "_get_nvidia_driver_version", return_value="530.41"):
            runtime_utils.apply_video_renderer_fallback(args, min_rtx_driver="535.129")
            first_pass = args.kit_args
            runtime_utils.apply_video_renderer_fallback(args, min_rtx_driver="535.129")

        self.assertIn("--/renderer/enabled=pxr", first_pass)
        self.assertEqual(args.kit_args, first_pass)

    def test_extract_rgb_frame_handles_batched_render(self):
        frame = np.arange(1 * 3 * 4 * 4, dtype=np.uint8).reshape(1, 3, 4, 4)

        rgb = runtime_utils._extract_rgb_frame(frame)

        self.assertEqual(rgb.shape, (3, 4, 3))

    def test_validate_rgb_rendering_accepts_non_black_frame(self):
        env = _FakeRenderEnv(
            [
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.array(
                    [
                        [[10, 20, 30], [40, 60, 80]],
                        [[90, 120, 150], [180, 210, 240]],
                    ],
                    dtype=np.uint8,
                ),
            ]
        )

        ok, reason = runtime_utils.validate_rgb_rendering(env, max_checks=4)

        self.assertTrue(ok)
        self.assertIn("valid frame found", reason)
        self.assertEqual(env.reset_calls, 1)
        self.assertGreaterEqual(env.step_calls, 1)

    def test_validate_rgb_rendering_reports_render_exception(self):
        env = _FakeRenderEnv([], render_error=RuntimeError("boom"))

        ok, reason = runtime_utils.validate_rgb_rendering(env, max_checks=2)

        self.assertFalse(ok)
        self.assertIn("env.render() failed", reason)
