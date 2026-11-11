import unittest
from unittest.mock import patch

import numpy as np

import GA.GA as GA
import GA.fitness_function as f


class TestTemporalGA(unittest.TestCase):
    def test_init_population(self) -> None:
        population = GA.init_population(10, seed=42, initial_std=0.01)
        np.testing.assert_array_equal(
            population, GA.init_population(10, seed=42, initial_std=0.01)
        )
        for index in [0, -1]:
            np.testing.assert_array_equal(
                population[:, index], np.broadcast_to(GA.SPREAD, (10, 8, 3))
            )
        error = population[:, 1:-1] - f.INTERPOLATED_REFERNCE_GAIT[1:-1]
        self.assertAlmostEqual(np.std(error), 0.01, delta=0.001)
        exact = GA.init_population(2, seed=42, initial_std=0)
        np.testing.assert_array_equal(exact, np.broadcast_to(f.INTERPOLATED_REFERNCE_GAIT, exact.shape))
        uniform = GA.init_population(10, seed=42, initial_std=None)
        np.testing.assert_array_equal(GA.init_population(10, seed=42), uniform)
        self.assertGreater(np.std(np.diff(uniform[:, 1:-1], axis=1)), 0.5)
        for invalid in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                GA.init_population(2, initial_std=invalid)
        self.assertTrue((population >= GA.MIN_JOINT_ANGLES).all())
        self.assertTrue((population <= GA.MAX_JOINT_ANGLES).all())

    def test_crossover(self) -> None:
        parents = np.broadcast_to(
            np.array([[-0.2, -0.4, -0.6], [0.2, 0.4, 0.6], [0.0, 0.0, 0.0]])[
                :, None, None, :
            ], (3, *f.GAIT_SHAPE)
        ).copy()
        original = parents.copy()
        np.testing.assert_array_equal(GA.crossover(parents, 0), parents)
        with patch("numpy.random.default_rng", return_value=np.random.default_rng(42)):
            children = GA.crossover(parents, 1)
        np.testing.assert_array_equal(children[-1], parents[-1])
        self.assertFalse(np.array_equal(children[:2], parents[:2]))
        self.assertTrue((children[:2] > parents[0]).all())
        self.assertTrue((children[:2] < parents[1]).all())
        # One mixing weight for the whole pair preserves these joint ratios
        # and introduces no frame-to-frame jumps between constant parents.
        np.testing.assert_allclose(children[..., 1], 2 * children[..., 0])
        np.testing.assert_allclose(children[..., 2], 3 * children[..., 0])
        self.assertEqual(np.ptp(children[0, ..., 0]), 0)
        np.testing.assert_allclose(np.diff(children, axis=1), 0)
        np.testing.assert_allclose(children[0] + children[1], parents[0] + parents[1], atol=1e-15)
        np.testing.assert_array_equal(parents, original)
        paired = np.tile(parents[:2], (8, 1, 1, 1))
        with patch("numpy.random.default_rng", return_value=np.random.default_rng(42)):
            mixed = GA.crossover(paired, 0.5)
        unchanged = np.all(mixed == paired, axis=(1, 2, 3))
        self.assertTrue(unchanged.any() and (~unchanged).any())
        np.testing.assert_array_equal(unchanged[::2], unchanged[1::2])
        for invalid in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                GA.crossover(parents, invalid)

    def test_mutate(self) -> None:
        population = np.zeros((2, *f.GAIT_SHAPE))
        np.testing.assert_array_equal(GA.mutate(population, 0, 0.1), population)
        np.testing.assert_array_equal(GA.mutate(population, 1, 0), population)
        mutated = GA.mutate(population, 1, 0.1)
        np.testing.assert_array_equal(mutated[:, [0, -1]], population[:, [0, -1]])
        self.assertGreater(np.std(mutated), 0.01)
        self.assertAlmostEqual(np.std(mutated[:, 1:-1]), 0.1, delta=0.005)
        sparse = GA.mutate(population, 0.1, 0.1)
        changed_fraction = np.count_nonzero(sparse) / population[:, 1:-1].size
        self.assertAlmostEqual(changed_fraction, 0.1, delta=0.02)
        self.assertTrue((mutated >= GA.MIN_JOINT_ANGLES).all())
        self.assertTrue((mutated <= GA.MAX_JOINT_ANGLES).all())
        self.assertTrue((population == 0).all())
        # With no random offset, a selected angle takes its neighbours' mean.
        # Read neighbours from the original population, not earlier updates.
        impulse = population.copy()
        impulse[:, 100, 0, 0] = 0.6
        averaged = GA.mutate(impulse, 1, 0)
        expected = population.copy()
        expected[:, [99, 101], 0, 0] = 0.3
        np.testing.assert_array_equal(averaged, expected)
        self.assertTrue((impulse[:, 100, 0, 0] == 0.6).all())
        for rate, std in [(-1, 0.1), (2, 0.1), (0.1, -1), (0.1, np.inf)]:
            with self.assertRaises(ValueError):
                GA.mutate(population, rate, std)

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
        default = GA.run_GA_training(2, 0, 0, 0, 0, GA.tournament_selection, 42)
        np.testing.assert_array_equal(default[0], GA.init_population(2, seed=42, initial_std=None))
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
        gait[:, f.RIGHT_MIRRORED] = [-0.1, 0.1, 0.1]
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
            f.start_end_pose_penalty(gait),
        ]
        score = f.fitness_function(gait, target)
        tolerances = {
            name: 2 * getattr(f, name)
            for name in [
                "TARGET_ANGLE_TOLERANCE", "STEP_SIZE_TOLERANCE",
                "ACCELERATION_TOLERANCE", "SYMMETRY_TOLERANCE",
                "START_END_TOLERANCE",
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
                    f.start_end_pose_penalty(gait),
                ],
            )
            self.assertAlmostEqual(
                f.fitness_function(gait, target), f.BASE_FITNESS - (f.BASE_FITNESS - score) / 2
            )
