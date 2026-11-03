import unittest

import numpy as np

from fitness_function import ANGLE_RANGES, GAIT_SHAPE, TARGET_STEP_SIZES, step_size_penalty


class TestStepSizePenalty(unittest.TestCase):
    def test_step_size_penalty(self):
        for scale in [0.0, 1.0, -1.0, 2.0]:
            with self.subTest(scale=scale):
                # Alternating poses exercise both directions within joint limits.
                gait = np.zeros(GAIT_SHAPE)
                gait[1::2] = scale * TARGET_STEP_SIZES
                expected = np.mean(
                    ((abs(scale) - 1) * TARGET_STEP_SIZES / ANGLE_RANGES) ** 2
                )
                result = step_size_penalty(gait)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected, delta=1e-15)

        for invalid in [np.zeros((1, 8, 3)), np.nan, np.inf, 10.0]:
            with self.subTest(invalid=invalid):
                gait = invalid if isinstance(invalid, np.ndarray) else np.full(GAIT_SHAPE, invalid)
                with self.assertRaises(ValueError):
                    step_size_penalty(gait)
