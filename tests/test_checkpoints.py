from __future__ import annotations

import os
import tempfile
import unittest

from so101_hackathon.utils.checkpoints import resolve_checkpoint_path


class CheckpointResolutionTests(unittest.TestCase):
    def test_direct_checkpoint_path_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pt")
            with open(checkpoint, "w", encoding="utf-8") as handle:
                handle.write("checkpoint")

            resolved = resolve_checkpoint_path(tmpdir, ".*", checkpoint)

            self.assertEqual(resolved, os.path.abspath(checkpoint))

    def test_regex_resolution_selects_latest_matching_run_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_a = os.path.join(tmpdir, "2026-04-01_12-00-00")
            run_b = os.path.join(tmpdir, "2026-04-02_12-00-00")
            os.makedirs(run_a)
            os.makedirs(run_b)
            with open(os.path.join(run_a, "model_10.pt"), "w", encoding="utf-8") as handle:
                handle.write("older")
            with open(os.path.join(run_b, "model_20.pt"), "w", encoding="utf-8") as handle:
                handle.write("newer")

            resolved = resolve_checkpoint_path(tmpdir, r"2026-04-.*", r"model_.*\.pt")

            self.assertTrue(resolved.endswith("2026-04-02_12-00-00/model_20.pt"))


if __name__ == "__main__":
    unittest.main()
