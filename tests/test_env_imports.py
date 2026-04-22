from __future__ import annotations

import unittest


class EnvImportBoundaryTests(unittest.TestCase):
    def test_lightweight_env_modules_import_without_isaac_runtime(self):
        """Verify lightweight env modules import without isaac runtime."""
        import so101_hackathon.envs as envs
        from so101_hackathon.envs import base_env, env_runtime, teleop_env

        self.assertEqual(envs.__all__, [])
        self.assertTrue(hasattr(base_env, "BaseHackathonEnvBuilder"))
        self.assertTrue(
            hasattr(env_runtime, "dynamic_reset_gripper_effort_limit_sim"))
        self.assertTrue(hasattr(teleop_env, "launch_and_make_teleop_env"))
