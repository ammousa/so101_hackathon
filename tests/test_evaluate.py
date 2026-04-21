from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.utils.eval_utils import (
    build_evaluation_payload,
    checkpoint_run_dir,
    evaluate_controller,
    log_evaluation_metrics,
    resolve_evaluation_output_dir,
    write_evaluation_config,
    write_summary_json,
)

import scripts.evaluate as evaluate_script


class FakeController(BaseController):
    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs):
        self.act_calls += 1
        return [0.0] * 6


class FakeEnv:
    def __init__(self):
        self.step_dt = 0.0
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        return [0.0] * 30

    def step(self, action):
        del action
        self.episode_step += 1
        done = self.episode_step >= 2
        return (
            [0.0] * 30,
            1.0,
            done,
            {
                "metrics": {
                    "joint_error": [0.1] * 6,
                    "action_rate": 0.25,
                    "ee_position_error": 0.2,
                    "ee_orientation_error": 0.3,
                    "invalid_state": False,
                    "failure": False,
                }
            },
        )


def context_manager_with(value):
    class _ContextManager:
        def __enter__(self):
            return value

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    return _ContextManager()


class EvaluateUtilsTests(unittest.TestCase):
    def test_checkpoint_run_dir_returns_checkpoint_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model_100.pt")
            with open(checkpoint, "w", encoding="utf-8") as handle:
                handle.write("checkpoint")

            self.assertEqual(checkpoint_run_dir(checkpoint), tmpdir)

    def test_resolve_evaluation_output_dir_nests_under_checkpoint_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model_100.pt")
            with open(checkpoint, "w", encoding="utf-8") as handle:
                handle.write("checkpoint")

            with mock.patch("so101_hackathon.utils.eval_utils.datetime") as mocked_datetime:
                mocked_datetime.now.return_value.strftime.return_value = "2026-04-19_10-11-12"
                output_dir = resolve_evaluation_output_dir(
                    controller_name="ppo",
                    requested_output_dir=None,
                    checkpoint_path=checkpoint,
                )

            self.assertEqual(output_dir, os.path.join(tmpdir, "evaluation", "2026-04-19_10-11-12"))

    def test_resolve_evaluation_output_dir_prefers_explicit_output_dir(self):
        resolved = resolve_evaluation_output_dir(
            controller_name="pd",
            requested_output_dir="custom/eval",
            checkpoint_path="/tmp/model.pt",
        )
        self.assertTrue(resolved.endswith(os.path.join("custom", "eval")))

    def test_write_evaluation_config_persists_args_and_configs(self):
        args = argparse.Namespace(controller="pd", seed=42, num_episodes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = write_evaluation_config(
                output_dir=tmpdir,
                args=args,
                env_config={"delay_steps": 2},
                controller_config={"kp": 0.5},
            )
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = handle.read()

        self.assertIn('"controller": "pd"', payload)
        self.assertIn('"delay_steps": 2', payload)
        self.assertIn('"kp": 0.5', payload)

    def test_build_and_write_summary_json(self):
        result = mock.Mock(metrics={"eval/joint_rmse": 0.12}, num_steps=17)
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = os.path.join(tmpdir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, "episode-0.mp4")
            with open(video_path, "w", encoding="utf-8") as handle:
                handle.write("video")

            payload = build_evaluation_payload(
                controller_name="pd",
                output_dir=tmpdir,
                config_path=os.path.join(tmpdir, "config.json"),
                result=result,
                video_dir=video_dir,
                include_video=True,
            )
            summary_path = write_summary_json(tmpdir, payload)
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = handle.read()

        self.assertIn('"controller": "pd"', summary)
        self.assertIn('"eval/joint_rmse": 0.12', summary)
        self.assertIn('"video"', summary)

    def test_log_evaluation_metrics_writes_tensorboard_scalars(self):
        summary_writer = mock.Mock()
        tensorboard_module = mock.Mock(SummaryWriter=mock.Mock(return_value=summary_writer))
        result = mock.Mock(metrics={"eval/joint_rmse": 0.12}, num_steps=17)

        with mock.patch.dict(sys.modules, {"torch.utils.tensorboard": tensorboard_module}):
            log_evaluation_metrics(
                controller_name="ppo",
                output_dir="/tmp/eval_logs",
                result=result,
                controller_config={"checkpoint_path": "/tmp/model.pt"},
            )

        tensorboard_module.SummaryWriter.assert_called_once()
        summary_writer.add_scalar.assert_any_call("eval/joint_rmse", 0.12, 17)
        summary_writer.add_text.assert_any_call("eval/controller", "ppo", 17)
        summary_writer.add_text.assert_any_call("eval/checkpoint_path", "/tmp/model.pt", 17)
        summary_writer.close.assert_called_once()


class EvaluateScriptTests(unittest.TestCase):
    def test_parser_exposes_single_checkpoint_path_argument(self):
        parser = evaluate_script.build_parser()
        args = parser.parse_args(["--checkpoint-path", "/tmp/model.pt"])

        self.assertEqual(args.checkpoint_path, "/tmp/model.pt")
        self.assertFalse(hasattr(args, "load_run"))
        self.assertFalse(hasattr(args, "load_checkpoint"))


class EvaluateControllerTests(unittest.TestCase):
    def test_evaluate_controller_runs_multiple_episodes(self):
        env = FakeEnv()
        controller = FakeController()

        result = evaluate_controller(env, controller, num_episodes=2)

        self.assertEqual(result.num_steps, 4)
        self.assertEqual(controller.reset_calls, 2)
        self.assertEqual(controller.act_calls, 4)
        self.assertAlmostEqual(result.metrics["eval/joint_rmse"], 0.1, places=6)
        self.assertAlmostEqual(result.metrics["eval/command_smoothness"], 0.25, places=6)
        self.assertEqual(result.metrics["eval/num_failures"], 0.0)
        self.assertEqual(result.metrics["eval/num_episodes"], 2.0)

    def test_progress_bar_updates_per_step(self):
        env = FakeEnv()
        controller = FakeController()
        progress = mock.Mock()

        with mock.patch(
            "so101_hackathon.utils.eval_utils.evaluation_progress_bar",
            return_value=context_manager_with(progress),
        ):
            result = evaluate_controller(env, controller, num_episodes=2, show_progress=True)

        self.assertEqual(result.num_steps, 4)
        self.assertEqual(progress.update.call_count, 4)
        progress.update.assert_called_with(1)
        self.assertGreaterEqual(progress.set_postfix.call_count, 4)
