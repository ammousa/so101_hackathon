from __future__ import annotations

import unittest

from so101_hackathon.utils.rl_utils import (
    TELEOP_HISTORY_LENGTH,
    TELEOP_JOINT_NAMES,
    TELEOP_TERM_ORDER,
    parse_teleop_observation,
)


class ObservationParsingTests(unittest.TestCase):
    def test_parse_single_observation_returns_latest_term_slices(self):
        joint_dim = len(TELEOP_JOINT_NAMES)
        term_size = joint_dim * TELEOP_HISTORY_LENGTH
        observation = list(range(len(TELEOP_TERM_ORDER) * term_size))

        parsed = parse_teleop_observation(observation)

        self.assertEqual(parsed["joint_names"], TELEOP_JOINT_NAMES)
        self.assertEqual(parsed["joint_dim"], joint_dim)
        self.assertEqual(parsed["history_length"], TELEOP_HISTORY_LENGTH)
        self.assertEqual(parsed["leader_joint_pos"],
                         observation[term_size - joint_dim: term_size])
        error_vel_start = 3 * term_size
        self.assertEqual(
            parsed["joint_error_vel"],
            observation[error_vel_start + term_size -
                        joint_dim: error_vel_start + term_size],
        )

    def test_parse_batched_observation_preserves_batch_dimension(self):
        joint_dim = len(TELEOP_JOINT_NAMES)
        term_size = joint_dim * TELEOP_HISTORY_LENGTH
        first = list(range(len(TELEOP_TERM_ORDER) * term_size))
        second = [value + 1000 for value in first]

        parsed = parse_teleop_observation([first, second])

        self.assertEqual(parsed["previous_action"][0], first[-joint_dim:])
        self.assertEqual(parsed["previous_action"][1], second[-joint_dim:])

    def test_parse_dict_observation_uses_policy_key(self):
        joint_dim = len(TELEOP_JOINT_NAMES)
        term_size = joint_dim * TELEOP_HISTORY_LENGTH
        observation = list(range(len(TELEOP_TERM_ORDER) * term_size))

        parsed = parse_teleop_observation({"policy": observation})

        self.assertEqual(parsed["leader_joint_pos"],
                         observation[term_size - joint_dim: term_size])

    def test_parse_rejects_wrong_observation_size(self):
        with self.assertRaises(ValueError):
            parse_teleop_observation([0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
