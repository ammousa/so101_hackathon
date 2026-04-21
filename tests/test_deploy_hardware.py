from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from so101_hackathon.deploy.hardware import _resolve_calibration_path


class DeployHardwareTests(unittest.TestCase):
    def test_resolve_calibration_path_uses_exact_device_id_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_dir = Path(tmpdir)
            target = calibration_dir / "my_awesome_leader_arm.json"
            target.write_text("{}", encoding="utf-8")

            resolved = _resolve_calibration_path(calibration_dir, "my_awesome_leader_arm", "leader")

            self.assertEqual(resolved, target)

    def test_resolve_calibration_path_falls_back_to_common_leader_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_dir = Path(tmpdir)
            target = calibration_dir / "leader.json"
            target.write_text("{}", encoding="utf-8")

            resolved = _resolve_calibration_path(calibration_dir, "my_awesome_leader_arm", "leader")

            self.assertEqual(resolved, target)


if __name__ == "__main__":
    unittest.main()
