from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from so101_hackathon.utils.rl_utils import build_training_log_dir

import scripts.deploy.calibrate_hardware as calibrate_script
import scripts.train_rl as train_script


class _FakeRunnerCfg:
    def __init__(self, *, resume: bool):
        self.experiment_name = "so101_hackathon_teleop"
        self.run_name = "testrun"
        self.device = "cpu"
        self.logger = "tensorboard"
        self.resume = resume
        self.load_run = ".*"
        self.load_checkpoint = ".*\\.pt"
        self.max_iterations = 123

    def to_dict(self):
        return {"experiment_name": self.experiment_name, "device": self.device}


class TrainScriptTests(unittest.TestCase):
    def test_build_training_log_dir_appends_run_name(self):
        fake_datetime = mock.Mock()
        fake_datetime.now.return_value.strftime.return_value = "2026-04-21_12-00-00"

        with mock.patch("so101_hackathon.utils.rl_utils.train_utils.datetime", fake_datetime):
            log_dir = build_training_log_dir("/tmp/logs", "baseline")

        self.assertEqual(log_dir, "/tmp/logs/2026-04-21_12-00-00_baseline")

    def test_train_main_runs_resume_flow_and_closes_resources(self):
        fake_env = mock.Mock(name="env")
        fake_app = mock.Mock(name="simulation_app")
        fake_runner = mock.Mock(name="runner")
        fake_runner_cls = mock.Mock(return_value=fake_runner)
        fake_launch = types.SimpleNamespace(env=fake_env, simulation_app=fake_app)
        fake_cfg = _FakeRunnerCfg(resume=True)
        fake_train_env_module = types.ModuleType("so101_hackathon.envs.teleop_env")
        fake_train_env_module.launch_and_make_teleop_env = mock.Mock(return_value=fake_launch)
        fake_runner_module = types.ModuleType("so101_hackathon.rl_training.on_policy_runner")
        fake_runner_module.OnPolicyRunner = fake_runner_cls

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(train_script, "normalize_device_for_runtime", return_value=("cpu", False)):
                with mock.patch.object(train_script, "build_teleop_ppo_runner_cfg", return_value=fake_cfg):
                    with mock.patch.object(train_script, "build_training_log_dir", return_value=str(Path(tmpdir) / "run")):
                        with mock.patch("so101_hackathon.utils.rl_utils.resolve_checkpoint_path", return_value="/tmp/model.pt"):
                            with mock.patch.dict(
                                "sys.modules",
                                {
                                    "so101_hackathon.envs.teleop_env": fake_train_env_module,
                                    "so101_hackathon.rl_training.on_policy_runner": fake_runner_module,
                                },
                            ):
                                result = train_script.main(
                                    [
                                        "--resume",
                                        "--checkpoint",
                                        "latest.pt",
                                        "--run-name",
                                        "testrun",
                                    ]
                                )

        self.assertIsNone(result)
        fake_train_env_module.launch_and_make_teleop_env.assert_called_once()
        fake_runner_cls.assert_called_once()
        fake_runner.load.assert_called_once_with("/tmp/model.pt")
        fake_runner.learn.assert_called_once_with(
            num_learning_iterations=123,
            init_at_random_ep_len=True,
        )
        fake_env.close.assert_called_once()
        fake_app.close.assert_called_once()

    def test_train_main_raises_helpful_runtime_error_when_runner_missing(self):
        with mock.patch.object(train_script, "normalize_device_for_runtime", return_value=("cpu", False)):
            with mock.patch.dict("sys.modules", {}, clear=False):
                with self.assertRaises(RuntimeError):
                    train_script.main([])


class CalibrationScriptTests(unittest.TestCase):
    def test_main_uses_role_specific_defaults_for_leader(self):
        with mock.patch.object(calibrate_script, "calibrate_so101_arm", return_value="/tmp/leader.json") as mocked_calibrate:
            result = calibrate_script.main(["--role", "leader"])

        self.assertEqual(result, 0)
        mocked_calibrate.assert_called_once_with(
            role="leader",
            port=calibrate_script.DEFAULT_LEADER_PORT,
            device_id=calibrate_script.DEFAULT_LEADER_ID,
            calibration_dir=str(calibrate_script.DEFAULT_CALIBRATION_DIR),
            disable_gripper=False,
        )

    def test_main_for_follower_passes_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(calibrate_script, "calibrate_so101_arm", return_value="/tmp/follower.json") as mocked_calibrate:
                result = calibrate_script.main(
                    [
                        "--role",
                        "follower",
                        "--port",
                        "/dev/ttyUSB9",
                        "--id",
                        "lab-arm",
                        "--calibration-dir",
                        tmpdir,
                        "--disable-gripper",
                    ]
                )

        self.assertEqual(result, 0)
        mocked_calibrate.assert_called_once_with(
            role="follower",
            port="/dev/ttyUSB9",
            device_id="lab-arm",
            calibration_dir=tmpdir,
            disable_gripper=True,
        )
