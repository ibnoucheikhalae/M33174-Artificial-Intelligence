import unittest
from unittest.mock import patch

import numpy as np

import GA
import fitness_function as f


class TestTemporalGA(unittest.TestCase):
    def test_init_population(self) -> None:
        population = GA.init_population(10, seed=42)
        np.testing.assert_array_equal(population, GA.init_population(10, seed=42))
        for index in [0, -1]:
            np.testing.assert_array_equal(
                population[:, index], np.broadcast_to(GA.SPREAD, (10, 8, 3))
            )
        error = population[:, 1:-1] - f.INTERPOLATED_REFERNCE_GAIT[1:-1]
        self.assertAlmostEqual(np.std(error), 0.01, delta=0.001)
        exact = GA.init_population(2, seed=42, initial_std=0)
        np.testing.assert_array_equal(exact, np.broadcast_to(f.INTERPOLATED_REFERNCE_GAIT, exact.shape))
        uniform = GA.init_population(10, seed=42, initial_std=None)
        self.assertGreater(np.std(np.diff(uniform[:, 1:-1], axis=1)), 0.5)
        for invalid in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                GA.init_population(2, initial_std=invalid)
        self.assertTrue((population >= GA.MIN_JOINT_ANGLES).all())
        self.assertTrue((population <= GA.MAX_JOINT_ANGLES).all())

    def test_roulette_selection(self) -> None:
        population = np.zeros((2, *f.GAIT_SHAPE))
        population[1] = 0.1
        selected = GA.roulette_selection(population, np.array([-2.0, -1.0]), 500)
        self.assertGreater(np.count_nonzero(selected[:, 0, 0, 0]), 350)
        selected = GA.roulette_selection(population, np.zeros(2), 10)
        self.assertTrue(np.isfinite(selected).all())
        self.assertTrue(np.isin(selected, [0.0, 0.1]).all())

    def test_tournament_selection(self) -> None:
        population = np.zeros((4, *f.GAIT_SHAPE))
        population[:, 1, 0, 0] = np.arange(4)
        fitness = np.arange(4, dtype=float)
        selected = GA.tournament_selection(population, fitness, 100, tournament_size=100)
        self.assertTrue((selected[:, 1, 0, 0] == 3).all())
        selected = GA.tournament_selection(population, fitness, 100, tournament_size=1)
        self.assertGreater(len(np.unique(selected[:, 1, 0, 0])), 1)
        for invalid in [0, -1]:
            with self.assertRaises(ValueError):
                GA.tournament_selection(population, fitness, 10, tournament_size=invalid)

    def test_run_GA_training(self) -> None:
        exact = GA.run_GA_training(2, 0, 0, 0, 0, GA.tournament_selection, initial_std=0)
        np.testing.assert_array_equal(
            exact[0], np.broadcast_to(f.INTERPOLATED_REFERNCE_GAIT, exact[0].shape)
        )
        for selection in [GA.tournament_selection, GA.roulette_selection]:
            with self.subTest(selection=selection.__name__):
                generations = GA.run_GA_training(8, 3, 1, 1, 0.1, selection, 42)
                scores = [GA.evaluate_fitness(population).max() for population in generations]
                self.assertTrue((np.diff(scores) >= 0).all())
                for index in [0, -1]:
                    np.testing.assert_array_equal(
                        generations[:, :, index], np.broadcast_to(GA.SPREAD, (4, 8, 8, 3))
                    )

    def test_fitness_function_rejects_jitter(self) -> None:
        reference = f.INTERPOLATED_REFERNCE_GAIT
        jittery = reference.copy()
        jittery[1::2, :, 0] += 0.05
        # Repeated three-degree hip jumps must not score close to the reference.
        self.assertLess(f.fitness_function(jittery), f.fitness_function(reference) - 0.5)

    def test_acceleration_penalty(self) -> None:
        gait = np.zeros(f.GAIT_SHAPE)
        self.assertEqual(f.acceleration_penalty(gait), 0.0)
        gait[1::2] = f.ACCELERATION_TOLERANCE / 2
        # Alternating positions give second differences of +/- one tolerance.
        self.assertAlmostEqual(f.acceleration_penalty(gait), f.ACCELERATION_TOLERANCE)

    def test_symmetry_penalty(self) -> None:
        gait = np.zeros(f.GAIT_SHAPE)
        self.assertEqual(f.symmetry_penalty(gait), 0.0)
        gait[:, f.LEFT_LEGS] = f.SYMMETRY_TOLERANCE
        self.assertAlmostEqual(f.symmetry_penalty(gait), f.SYMMETRY_TOLERANCE)
        gait[:, f.RIGHT_MIRRORED] = -f.SYMMETRY_TOLERANCE
        self.assertEqual(f.symmetry_penalty(gait), 0.0)

    def test_fitness_function_scales_raw_errors(self) -> None:
        gait = np.zeros(f.GAIT_SHAPE)
        gait[1::2] = 0.01
        target = np.zeros(f.GAIT_SHAPE)
        raw_errors = [
            f.target_gait_penalty(gait, target),
            f.step_size_penalty(gait, target),
            f.acceleration_penalty(gait),
            f.symmetry_penalty(gait),
        ]
        score = f.fitness_function(gait, target)
        tolerances = {
            name: 2 * getattr(f, name)
            for name in [
                "TARGET_ANGLE_TOLERANCE", "STEP_SIZE_TOLERANCE",
                "ACCELERATION_TOLERANCE", "SYMMETRY_TOLERANCE",
            ]
        }
        with patch.multiple(f, **tolerances):
            # Tolerances affect the combined score, never the raw measurements.
            self.assertEqual(
                raw_errors,
                [
                    f.target_gait_penalty(gait, target),
                    f.step_size_penalty(gait, target),
                    f.acceleration_penalty(gait),
                    f.symmetry_penalty(gait),
                ],
            )
            self.assertAlmostEqual(
                f.fitness_function(gait, target), f.BASE_FITNESS - (f.BASE_FITNESS - score) / 2
            )
