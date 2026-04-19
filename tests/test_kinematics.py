from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    torch = None

if torch is not None:
    from so101_hackathon.sim.kinematics import (
        compute_so101_chain_points,
        compute_so101_ee_jacobian,
        compute_so101_ee_position,
    )


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
TELEOP_JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


@unittest.skipIf(torch is None, "torch is required for kinematics tests")
class So101KinematicsTests(unittest.TestCase):
    def test_gripper_input_does_not_change_gripper_link_position(self):
        arm_joint_pos = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.2]], dtype=torch.float32)
        with_gripper_a = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.2, 0.0]], dtype=torch.float32)
        with_gripper_b = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.2, 0.9]], dtype=torch.float32)

        arm_position = compute_so101_ee_position(arm_joint_pos, joint_names=ARM_JOINT_NAMES)
        ee_position_a = compute_so101_ee_position(with_gripper_a, joint_names=TELEOP_JOINT_NAMES)
        ee_position_b = compute_so101_ee_position(with_gripper_b, joint_names=TELEOP_JOINT_NAMES)

        self.assertTrue(torch.allclose(arm_position, ee_position_a, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(ee_position_a, ee_position_b, atol=1e-6, rtol=1e-6))

    def test_analytic_position_jacobian_matches_finite_difference(self):
        joint_pos = torch.tensor([[0.15, -0.35, 0.25, -0.2, 0.1]], dtype=torch.float32)
        analytic = compute_so101_ee_jacobian(joint_pos, joint_names=ARM_JOINT_NAMES)

        epsilon = 1.0e-4
        finite_difference_columns = []
        base_position = compute_so101_ee_position(joint_pos, joint_names=ARM_JOINT_NAMES)
        for joint_index in range(joint_pos.shape[1]):
            perturbed = joint_pos.clone()
            perturbed[:, joint_index] += epsilon
            perturbed_position = compute_so101_ee_position(perturbed, joint_names=ARM_JOINT_NAMES)
            finite_difference_columns.append((perturbed_position - base_position) / epsilon)
        finite_difference = torch.stack(finite_difference_columns, dim=-1)

        self.assertTrue(torch.allclose(analytic, finite_difference, atol=2e-3, rtol=2e-3))

    def test_chain_points_include_all_arm_joints_plus_end_effector(self):
        joint_pos = torch.zeros((2, len(TELEOP_JOINT_NAMES)), dtype=torch.float32)
        chain_points = compute_so101_chain_points(joint_pos, joint_names=TELEOP_JOINT_NAMES)

        self.assertEqual(tuple(chain_points.shape), (2, len(ARM_JOINT_NAMES) + 1, 3))


if __name__ == "__main__":
    unittest.main()
