import unittest

import numpy as np

import GA.fitness_function as f
from GA.GA import evaluate_fitness
from GA.plot_spider_pose import forward_leg_kinematics2


class TestTargetGait(unittest.TestCase):
    def test_reference_matches_renderer_convention(self):
        base_angles = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
        for pose in [f.SPREAD, f.CLOSE, f.INTERPOLATED_REFERNCE_GAIT[75]]:
            feet = []
            for angle, joints in zip(base_angles, pose):
                base = [1.5 * np.cos(angle), np.sin(angle), 0]
                feet.append(forward_leg_kinematics2(base, angle, joints, [1.2, 0.7, 1.0])[-1])
            feet = np.asarray(feet)
            np.testing.assert_allclose(
                feet[:4], feet[[7, 6, 5, 4]] * [1, -1, 1], atol=1e-12
            )
        self.assertEqual(f.symmetry_penalty(f.INTERPOLATED_REFERNCE_GAIT), 0.0)

    def test_target_gait_penalty(self):
        target = f.INTERPOLATED_REFERNCE_GAIT
        self.assertEqual(f.target_gait_penalty(target), 0.0)
        gait = target.copy()
        gait[42, 0, 0] += 0.1  # An error away from the keyframes must count too.
        expected = 0.1 / np.sqrt(gait.size)
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
        # Custom targets still incur the independent SPREAD endpoint requirement.
        gait = np.zeros(f.GAIT_SHAPE)
        expected = f.BASE_FITNESS - (
            f.START_END_WEIGHT / f.START_END_TOLERANCE
        ) * f.start_end_pose_penalty(gait)
        self.assertEqual(f.fitness_function(gait, gait), expected)

    def test_start_end_pose_penalty(self):
        gait = np.broadcast_to(f.SPREAD, f.GAIT_SHAPE).copy()
        self.assertEqual(f.start_end_pose_penalty(gait), 0.0)
        for frame in [0, f.EDGE_FRAMES - 1, -f.EDGE_FRAMES, -1]:
            with self.subTest(frame=frame):
                changed = gait.copy()
                changed[frame, 0, 0] += 0.1
                self.assertAlmostEqual(
                    f.start_end_pose_penalty(changed),
                    0.1 / (3.14 * 2 * f.EDGE_FRAMES * 24),
                )
        gait[f.EDGE_FRAMES:-f.EDGE_FRAMES, 0, 0] += 0.1
        self.assertEqual(f.start_end_pose_penalty(gait), 0.0)
        for invalid in [
            np.zeros((3, 8, 3)),
            np.full(f.GAIT_SHAPE, np.nan),
            np.full(f.GAIT_SHAPE, np.inf),
            np.full(f.GAIT_SHAPE, 10.0),
        ]:
            with self.assertRaises(ValueError):
                f.start_end_pose_penalty(invalid)
