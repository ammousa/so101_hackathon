from __future__ import annotations

import math
import unittest

from so101_hackathon.deploy.runtime import parse_joint_limits_from_urdf
from so101_hackathon.sim.robots import SO101_JOINT_NAMES
from so101_hackathon.sim.robots.so101_follower_spec import (
    SO101_FOLLOWER_ASSET_PATH,
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMITS_DEG,
    follower_joint_limits_rad_map,
    joint_radians_to_motor_value,
    motor_value_to_joint_radians,
)
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES


class RobotSpecTests(unittest.TestCase):
    def test_obs_utils_joint_names_follow_shared_robot_spec(self):
        """Verify obs utils joint names follow shared robot spec."""
        self.assertEqual(TELEOP_JOINT_NAMES, SO101_JOINT_NAMES)

    def test_joint_limit_lookup_uses_shared_robot_spec(self):
        """Verify joint limit lookup uses shared robot spec."""
        self.assertEqual(parse_joint_limits_from_urdf(),
                         follower_joint_limits_rad_map())

    def test_motor_joint_conversion_round_trip(self):
        """Verify motor joint conversion round trip."""
        for joint_name, (motor_lower, motor_upper) in SO101_FOLLOWER_MOTOR_LIMITS.items():
            midpoint = 0.5 * (motor_lower + motor_upper)
            radians_value = motor_value_to_joint_radians(joint_name, midpoint)
            recovered = joint_radians_to_motor_value(joint_name, radians_value)
            self.assertAlmostEqual(recovered, midpoint, places=6)

    def test_internal_robot_package_does_not_export_legacy_alias(self):
        """Verify internal robot package does not export legacy alias."""
        import so101_hackathon.sim.robots as robots

        self.assertFalse(hasattr(robots, "SO_ARM101_CFG"))

    def test_internalized_robot_asset_exists(self):
        """Verify internalized robot asset exists."""
        self.assertTrue(SO101_FOLLOWER_ASSET_PATH.is_file())

    def test_joint_limit_degrees_are_reasonable(self):
        """Verify joint limit degrees are reasonable."""
        elbow_lower, elbow_upper = SO101_FOLLOWER_USD_JOINT_LIMITS_DEG["elbow_flex"]
        self.assertLess(elbow_lower, elbow_upper)
        self.assertTrue(math.isfinite(elbow_lower))
        self.assertTrue(math.isfinite(elbow_upper))


if __name__ == "__main__":
    unittest.main()
