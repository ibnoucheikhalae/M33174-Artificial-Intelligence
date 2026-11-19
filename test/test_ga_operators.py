"""Tests for the genetic operators.

The key properties under test are the lecturer's constraints: operators never
clamp, never smooth, never pin frames -- constraint handling belongs to the
fitness function.
"""

import unittest
from unittest.mock import patch

import numpy as np

import GA.GA as GA
from GA import fitness_function as f


class TestInitPopulation(unittest.TestCase):
    def test_shape_and_determinism(self) -> None:
        population = GA.init_population(10, seed=42)
        self.assertEqual(population.shape, (10, 300, 8, 3))
        self.assertTrue(np.isfinite(population).all())
        np.testing.assert_array_equal(population, GA.init_population(10, seed=42))

    def test_initial_std_seeds_every_individual_with_gaussian_noise(self) -> None:
        reference = GA.sinusoidal_seed_gait()
        for std in (0.0, 0.01, 0.1):
            with self.subTest(std=std):
                population = GA.init_population(10, seed=42, initial_std=std)
                noise = population - reference
                self.assertAlmostEqual(float(noise.mean()), 0.0, delta=max(std * 0.02, 1e-15))
                self.assertAlmostEqual(float(noise.std()), std, delta=max(std * 0.02, 1e-15))
                if std == 0:
                    np.testing.assert_array_equal(
                        population, np.broadcast_to(reference, population.shape)
                    )
                else:
                    # Every individual starts near the same reference, including end frames.
                    self.assertTrue((np.std(noise, axis=(1, 2, 3)) > std * 0.95).all())
                    self.assertTrue((np.std(population[:, [0, -1]], axis=0) > 0).all())
        for invalid in (-0.1, np.inf, np.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GA.init_population(2, initial_std=invalid)
        self.assertEqual(GA.init_population(0).shape, (0, 300, 8, 3))

    def test_seed_gaits_use_the_same_step_count_for_hips_and_knees(self) -> None:
        for steps in (1, 2, 3):
            with self.subTest(steps=steps), patch.object(GA, "NUM_STEPS", steps):
                gait = GA.sinusoidal_seed_gait()
                # Count complete spread-close-spread excursions and lifts of L1.
                hip_closed = GA.ideal_hip_movement()[:, 0, 0] > -0.12
                knee_raised = gait[:, 0, 2] > GA.SPREAD[0, 2] + 0.1
                for active in (hip_closed, knee_raised):
                    self.assertEqual(
                        np.count_nonzero(np.diff(active.astype(int)) == 1), steps
                    )
                self.assertTrue((gait.std(axis=0) > 0).all())
                np.testing.assert_allclose(gait[0], gait[-1], atol=1e-12)
                self.assertFalse(np.allclose(gait[-1], GA.SPREAD))
                # The two tetrapod groups take turns lifting.
                lifts = gait[:, :, 2] - GA.SPREAD[:, 2]
                self.assertFalse(((lifts[:, 0] > 0.1) & (lifts[:, 1] > 0.1)).any())

    def test_seed_does_not_blend_back_to_standing_at_end(self) -> None:
        gait = GA.sinusoidal_seed_gait()
        # Near a natural hip turning point, frame-to-frame motion stays small.
        self.assertLess(float(np.abs(np.diff(gait[-20:, :, 0], axis=0)).max()), 0.003)
        self.assertFalse(np.allclose(gait[-1], GA.SPREAD))

    def test_seed_hips_sweep_forward_during_lift(self) -> None:
        gait = GA.sinusoidal_seed_gait()
        up = f.foot_heights(gait) > f.STANCE_Z + f.AIRBORNE_CLEARANCE
        forward = -np.sign(f.SOCKET_HEADINGS) * np.diff(gait[:, :, 0], axis=0)
        for leg in range(GA.NUM_LEGS):
            airborne_motion = forward[:, leg][up[:-1, leg] & up[1:, leg]]
            self.assertTrue((airborne_motion > 0).all())
            self.assertLess(float(forward[:, leg][~up[:-1, leg] & ~up[1:, leg]].sum()), 0)

    def test_run_GA_passes_initial_std_to_population(self) -> None:
        generations = GA.run_GA(4, 0, 0, 0, 0, GA.tournament_selection, 7, initial_std=0.02)
        np.testing.assert_array_equal(generations, GA.init_population(4, 7, initial_std=0.02))


class TestCrossover(unittest.TestCase):
    def test_children_inherit_whole_legs(self) -> None:
        parents = np.stack([
            np.zeros((300, 8, 3)),
            np.ones((300, 8, 3)),
        ])
        with patch("numpy.random.default_rng", return_value=np.random.default_rng(1)):
            children = GA.crossover(parents, 1.0)
        # Each leg trajectory is wholly 0s or wholly 1s: no mixing inside a leg.
        per_leg = children.reshape(2, 300, 8, 3)
        for child in per_leg:
            leg_values = child.transpose(1, 0, 2).reshape(8, -1)
            self.assertTrue(np.all((leg_values == 0).all(axis=1) | (leg_values == 1).all(axis=1)))
        # The two children partition the parents' legs between them.
        np.testing.assert_array_equal(per_leg[0] + per_leg[1], np.ones((300, 8, 3)))

    def test_zero_rate_is_identity(self) -> None:
        parents = np.random.default_rng(0).normal(size=(4, 300, 8, 3))
        np.testing.assert_array_equal(GA.crossover(parents, 0.0), parents)

    def test_no_clamping(self) -> None:
        # Parents far outside the joint limits pass through untouched.
        parents = np.full((2, 300, 8, 3), 10.0)
        children = GA.crossover(parents, 1.0)
        np.testing.assert_array_equal(children, parents)

    def test_invalid_rate_raises(self) -> None:
        parents = np.zeros((2, 300, 8, 3))
        for invalid in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                GA.crossover(parents, invalid)


class TestMutate(unittest.TestCase):
    def test_zero_rate_and_zero_std_are_identity(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        np.testing.assert_array_equal(GA.mutate(population, 0, 0.1), population)
        np.testing.assert_array_equal(GA.mutate(population, 1, 0), population)

    def test_mutation_is_pure_additive_gaussian(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        mutated = GA.mutate(population, 1, 0.1)
        # Every gene moved, with the requested spread, and the input is untouched.
        self.assertAlmostEqual(np.std(mutated), 0.1, delta=0.005)
        self.assertAlmostEqual(np.mean(mutated), 0.0, delta=0.005)
        self.assertTrue((population == 0).all())

    def test_first_and_last_frames_mutate_too(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        mutated = GA.mutate(population, 1, 0.1)
        for index in [0, -1]:
            self.assertGreater(np.abs(mutated[:, index]).max(), 0)

    def test_no_clamping_to_joint_limits(self) -> None:
        # Mutating from the upper limit must be able to exceed it.
        population = np.broadcast_to(
            f.MAX_JOINT_ANGLES, (2, 300, 8, 3)
        ).copy()
        mutated = GA.mutate(population, 1, 0.5)
        self.assertTrue((mutated > f.MAX_JOINT_ANGLES).any())

    def test_sparse_rate_mutates_expected_fraction(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        sparse = GA.mutate(population, 0.1, 0.1)
        changed_fraction = np.count_nonzero(sparse) / population.size
        self.assertAlmostEqual(changed_fraction, 0.1, delta=0.02)

    def test_invalid_parameters_raise(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        for rate, std in [(-1, 0.1), (2, 0.1), (0.1, -1), (0.1, np.inf)]:
            with self.assertRaises(ValueError):
                GA.mutate(population, rate, std)


class TestSelection(unittest.TestCase):
    def test_roulette_handles_negative_fitness(self) -> None:
        population = np.zeros((2, 300, 8, 3))
        population[1] = 0.1
        selected = GA.roulette_selection(population, np.array([-2.0, -1.0]), 500)
        self.assertGreater(np.count_nonzero(selected[:, 0, 0, 0]), 350)
        selected = GA.roulette_selection(population, np.zeros(2), 10)
        self.assertTrue(np.isfinite(selected).all())
        self.assertTrue(np.isin(selected, [0.0, 0.1]).all())

    def test_tournament_selection(self) -> None:
        population = np.zeros((4, 300, 8, 3))
        population[:, 1, 0, 0] = np.arange(4)
        fitness = np.arange(4, dtype=float)
        selected = GA.tournament_selection(population, fitness, 100, tournament_size=100)
        self.assertTrue((selected[:, 1, 0, 0] == 3).all())
        selected = GA.tournament_selection(population, fitness, 100, tournament_size=1)
        self.assertGreater(len(np.unique(selected[:, 1, 0, 0])), 1)
        for invalid in [0, -1]:
            with self.assertRaises(ValueError):
                GA.tournament_selection(population, fitness, 10, tournament_size=invalid)


class TestTraining(unittest.TestCase):
    def test_elitism_keeps_best_fitness_monotonic(self) -> None:
        for selection in [GA.tournament_selection, GA.roulette_selection]:
            with self.subTest(selection=selection.__name__):
                scores = []
                GA.run_GA(
                    8, 3, 1, 0.05, 0.02, selection, 42,
                    on_generation=lambda gen, fitness, pop: scores.append(fitness.max()),
                )
                self.assertTrue((np.diff(scores) >= -1e-12).all())

    def test_initial_generation_matches_init_population(self) -> None:
        generations = GA.run_GA(4, 0, 0, 0, 0, GA.tournament_selection, 7)
        np.testing.assert_array_equal(generations, GA.init_population(4, seed=7))


if __name__ == "__main__":
    unittest.main()
