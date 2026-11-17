"""Tests for the penalty-based fitness function.

The central property: invalid candidates (out-of-range angles, crossed legs)
are scored in proportion to the size of the violation, never rejected.
"""

import unittest

import numpy as np

import GA.GA as GA
from GA import fitness_function as f


def static_gait(pose) -> np.ndarray:
    return np.broadcast_to(pose, f.GAIT_SHAPE).copy()


class TestValidation(unittest.TestCase):
    def test_out_of_range_angles_are_accepted(self) -> None:
        gait = np.full(f.GAIT_SHAPE, 10.0)  # far outside every joint limit
        self.assertTrue(np.isfinite(f.fitness_function(gait)))

    def test_shape_and_nan_are_rejected(self) -> None:
        for invalid in [np.zeros((3, 8, 3)), np.full(f.GAIT_SHAPE, np.nan)]:
            with self.assertRaises(ValueError):
                f.fitness_function(invalid)


class TestJointLimitPenalty(unittest.TestCase):
    def test_zero_within_limits(self) -> None:
        self.assertEqual(f.joint_limit_penalty(static_gait(GA.SPREAD)), 0.0)

    def test_grows_with_violation_size(self) -> None:
        small = static_gait(GA.SPREAD)
        large = static_gait(GA.SPREAD)
        small[:, 0, 0] = f.MAX_JOINT_ANGLES[0, 0] + 0.1
        large[:, 0, 0] = f.MAX_JOINT_ANGLES[0, 0] + 1.0
        self.assertGreater(f.joint_limit_penalty(small), 0.0)
        self.assertAlmostEqual(
            f.joint_limit_penalty(large) / f.joint_limit_penalty(small), 10.0
        )

    def test_below_minimum_counts_too(self) -> None:
        gait = static_gait(GA.SPREAD)
        gait[:, 2, 2] = f.MIN_JOINT_ANGLES[2, 2] - 0.5
        self.assertGreater(f.joint_limit_penalty(gait), 0.0)


class TestLegOverlapPenalty(unittest.TestCase):
    def test_zero_for_task_poses(self) -> None:
        self.assertEqual(f.leg_overlap_penalty(static_gait(GA.SPREAD)), 0.0)
        self.assertEqual(f.leg_overlap_penalty(static_gait(GA.CLOSE)), 0.0)

    def test_crossed_neighbours_are_penalised(self) -> None:
        pose = GA.SPREAD.copy()
        # Swing L1 (base 45 deg) anticlockwise past L2 (base 75 deg).
        pose[0, 0] = 0.9  # 45 deg + ~51.5 deg > 75 deg + L2's hip
        gait = static_gait(pose)
        self.assertGreater(f.leg_overlap_penalty(gait), 0.0)

    def test_penalty_scales_with_crossing_depth(self) -> None:
        shallow, deep = GA.SPREAD.copy(), GA.SPREAD.copy()
        shallow[0, 0] = 0.5
        deep[0, 0] = 1.2
        self.assertGreater(
            f.leg_overlap_penalty(static_gait(deep)),
            f.leg_overlap_penalty(static_gait(shallow)),
        )


class TestTargetPosePenalty(unittest.TestCase):
    def test_zero_when_keyframes_match(self) -> None:
        gait = GA.ideal_hip_movement()
        self.assertAlmostEqual(f.target_pose_penalty(gait), 0.0)

    def test_only_keyframes_are_scored(self) -> None:
        gait = GA.ideal_hip_movement()
        gait[42] += 0.3  # not a keyframe
        self.assertAlmostEqual(f.target_pose_penalty(gait), 0.0)
        gait[150] += 0.3  # keyframe
        self.assertGreater(f.target_pose_penalty(gait), 0.0)


class TestAccelerationPenalty(unittest.TestCase):
    def test_constant_motion_is_free(self) -> None:
        gait = static_gait(GA.SPREAD)
        self.assertEqual(f.acceleration_penalty(gait), 0.0)

    def test_jitter_is_penalised(self) -> None:
        gait = static_gait(GA.SPREAD)
        gait[1::2] += 0.05
        self.assertGreater(f.acceleration_penalty(gait), 0.0)


class TestSymmetryPenalty(unittest.TestCase):
    def test_mirrored_gait_scores_zero(self) -> None:
        self.assertEqual(f.symmetry_penalty(static_gait(GA.SPREAD)), 0.0)

    def test_asymmetry_is_penalised(self) -> None:
        gait = static_gait(GA.SPREAD)
        gait[:, 0, 1] += 0.4  # raise L1's thigh only
        self.assertGreater(f.symmetry_penalty(gait), 0.0)


class TestCouplePenalty(unittest.TestCase):
    def test_standing_and_seed_couples_match(self) -> None:
        self.assertEqual(f.couple_penalty(static_gait(GA.SPREAD)), 0.0)
        self.assertEqual(f.couple_penalty(static_gait(GA.CLOSE)), 0.0)
        seed = GA.sinusoidal_seed_gait(np.random.default_rng(5))
        self.assertLess(f.couple_penalty(seed), 0.02)

    def test_broken_couple_is_penalised(self) -> None:
        gait = static_gait(GA.SPREAD)
        gait[:, 0, 1] += 0.4  # L1 thigh, L3 unchanged
        self.assertGreater(f.couple_penalty(gait), 0.0)

    def test_different_hips_do_not_count(self) -> None:
        # Front and rear hips must differ to hit CLOSE; that is not a couple break.
        self.assertEqual(f.couple_penalty(GA.ideal_hip_movement()), 0.0)


class TestMovementReward(unittest.TestCase):
    def test_static_gait_earns_nothing(self) -> None:
        self.assertAlmostEqual(f.movement_reward(static_gait(GA.SPREAD)), 0.0)

    def test_all_joints_moving_earns_full_credit(self) -> None:
        wave = np.sin(np.linspace(0, 4 * np.pi, f.NUM_FRAMES))
        gait = static_gait(GA.SPREAD) + wave[:, None, None]
        self.assertEqual(f.movement_reward(gait), 1.0)

    def test_one_stuck_joint_lowers_reward(self) -> None:
        wave = np.sin(np.linspace(0, 4 * np.pi, f.NUM_FRAMES))
        gait = static_gait(GA.SPREAD) + wave[:, None, None]
        gait[:, 3, 2] = GA.SPREAD[3, 2]  # freeze one knee
        self.assertAlmostEqual(f.movement_reward(gait), 23 / 24)


class TestStepReward(unittest.TestCase):
    def test_standing_gait_is_not_a_step(self) -> None:
        self.assertAlmostEqual(f.step_reward(static_gait(GA.SPREAD)), 0.0, delta=1e-3)
        self.assertAlmostEqual(f.step_reward(GA.ideal_hip_movement()), 0.0, delta=0.05)

    def test_seed_steps_harder_than_a_hip_slide(self) -> None:
        seed = GA.sinusoidal_seed_gait(np.random.default_rng(5))
        self.assertGreater(f.step_reward(seed), 0.4)
        self.assertGreater(f.step_reward(seed), f.step_reward(GA.ideal_hip_movement()) + 0.3)

    def test_all_eight_legs_in_the_air_is_penalised(self) -> None:
        airborne = static_gait(GA.SPREAD)
        airborne[:, :, 2] += 1.0  # raise every knee
        self.assertGreater(f.support_penalty(airborne), 0.5)
        self.assertEqual(f.support_penalty(static_gait(GA.SPREAD)), 0.0)


class TestCombinedFitness(unittest.TestCase):
    def test_breakdown_sums_to_fitness(self) -> None:
        gait = GA.sinusoidal_seed_gait(np.random.default_rng(5))
        breakdown = f.fitness_breakdown(gait)
        self.assertAlmostEqual(
            f.fitness_function(gait), f.BASE_FITNESS + sum(breakdown.values())
        )

    def test_invalid_candidate_scores_worse_not_errors(self) -> None:
        valid = GA.sinusoidal_seed_gait(np.random.default_rng(5))
        invalid = valid.copy()
        invalid[:, 0, 0] += 3.0
        self.assertLess(f.fitness_function(invalid), f.fitness_function(valid))


if __name__ == "__main__":
    unittest.main()
