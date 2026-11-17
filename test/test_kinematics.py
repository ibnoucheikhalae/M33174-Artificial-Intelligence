"""The 3D plot and the fitness scorer must agree on foot positions."""

import unittest

import numpy as np

import GA.GA as GA
from GA.kinematics import (
    SOCKET_HEADINGS,
    LEG_BASES,
    SEGMENT_LENGTHS,
    foot_positions,
    forward_leg_kinematics2,
)


class TestKinematics(unittest.TestCase):
    def test_one_leg_matches_batched_feet(self) -> None:
        pose = GA.SPREAD
        batched = foot_positions(pose[None])[0]
        for i, joints in enumerate(pose):
            foot = forward_leg_kinematics2(
                LEG_BASES[i], float(SOCKET_HEADINGS[i]), joints, SEGMENT_LENGTHS
            )[-1]
            np.testing.assert_allclose(batched[i], foot, atol=1e-12)

    def test_standing_feet_are_below_the_body(self) -> None:
        z = foot_positions(GA.SPREAD[None])[0, :, 2]
        self.assertTrue((z < -0.8).all())
