import unittest

import numpy as np

import fitness_function as f
from GA import evaluate_fitness


class TestTargetGait(unittest.TestCase):
    def test_target_gait_penalty(self):
        target = f.INTERPOLATED_REFERNCE_GAIT
        self.assertEqual(f.target_gait_penalty(target), 0.0)
        gait = target.copy()
        gait[42, 0, 0] += 0.1  # An error away from the keyframes must count too.
        expected = 0.1 / (f.ANGLE_RANGES[0, 0] * np.sqrt(gait.size))
        self.assertAlmostEqual(f.target_gait_penalty(gait), expected, delta=1e-15)
        self.assertEqual(f.target_gait_penalty(gait, gait), 0.0)

        for invalid in [np.zeros((3, 8, 3)), np.nan, np.inf, 10.0]:
            angles = invalid if isinstance(invalid, np.ndarray) else np.full(f.GAIT_SHAPE, invalid)
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    f.target_gait_penalty(angles)
                with self.assertRaises(ValueError):
                    f.target_gait_penalty(target, angles)

    def test_fitness_function_default_target(self):
        gait = f.INTERPOLATED_REFERNCE_GAIT
        score = f.fitness_function(gait)
        self.assertIsInstance(score, float)
        self.assertEqual(score, f.fitness_function(gait, gait))
        np.testing.assert_array_equal(evaluate_fitness(gait[None]), [score])

    def test_fitness_function_custom_target(self):
        # A static, symmetric gait matching its custom target has no penalties.
        gait = np.zeros(f.GAIT_SHAPE)
        self.assertEqual(f.fitness_function(gait, gait), f.BASE_FITNESS)
