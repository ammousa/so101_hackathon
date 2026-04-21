from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from so101_hackathon.utils.rl_utils import build_training_log_dir

import scripts.deploy.calibrate_hardware as calibrate_script


class TrainScriptTests(unittest.TestCase):
    def test_build_training_log_dir_appends_run_name(self):
        fake_datetime = mock.Mock()
        fake_datetime.now.return_value.strftime.return_value = "2026-04-21_12-00-00"

        with mock.patch("so101_hackathon.utils.rl_utils.train_utils.datetime", fake_datetime):
            log_dir = build_training_log_dir("/tmp/logs", "baseline")

        self.assertEqual(log_dir, "/tmp/logs/2026-04-21_12-00-00_baseline")

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
