import unittest

import numpy as np

from fitness_function import (
    ANGLE_RANGES,
    GAIT_SHAPE,
    INTERPOLATED_REFERNCE_GAIT,
    step_size_penalty,
)


class TestStepSizePenalty(unittest.TestCase):
    def test_step_size_penalty(self):
        self.assertEqual(step_size_penalty(INTERPOLATED_REFERNCE_GAIT), 0.0)
        target = np.zeros(GAIT_SHAPE)
        target[1::2] = 0.01
        for scale in [0.0, 1.0, -1.0, 2.0]:
            with self.subTest(scale=scale):
                # Alternating poses exercise both directions within joint limits.
                # Reversing the steps must incur an error, even at the same speed.
                gait = scale * target
                expected = np.mean(((scale - 1) * 0.01 / ANGLE_RANGES) ** 2)
                result = step_size_penalty(gait, target)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected, delta=1e-15)

        for invalid in [np.zeros((1, 8, 3)), np.nan, np.inf, 10.0]:
            with self.subTest(invalid=invalid):
                gait = invalid if isinstance(invalid, np.ndarray) else np.full(GAIT_SHAPE, invalid)
                with self.assertRaises(ValueError):
                    step_size_penalty(gait)
                with self.assertRaises(ValueError):
                    step_size_penalty(target, gait)
