from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_list_controllers_script_prints_builtin_names(self):
        result = self._run("scripts/list_controllers.py")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines(),
            ["pd", "ppo", "raw"],
        )

    def test_help_commands_work_without_heavy_runtime_deps(self):
        for script_path in ("scripts/train_rl.py", "scripts/evaluate.py", "scripts/play.py"):
            result = self._run(script_path, "--help")
            self.assertEqual(result.returncode, 0, msg=f"{script_path}: {result.stderr}")
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
