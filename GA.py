import math
import numpy as np
import sys
import random

from numpy.typing import ArrayLike, NDArray

NUM_FRAMES = 300
NUM_LEGS = 8
NUM_LEG_SEGS = 3

#todo: Update to whatever angles we decide on for this
MIN_JOINT_ANGLE = math.pi / 6  # 30deg
MAX_JOINT_ANGLE = math.pi / 2  # 90deg


rng = np.random.default_rng()

#! Assuming float64, we may be able to optimise this
def init_population(n: int) -> NDArray[np.float64]:
    """
    Initialise a population of n individuals
    The structure of each individual is described below:

    - 1 joint angle is a number - e.g., 45
    - 1 leg is 3 joint angles - e.g., (45, 90, 120)
    - 1 spider is 8 legs - e.g., ((45, 90, 120), * 8)
    - 1 individual is the spider's movement over 300 frames:
        [
            [(45, 90, 120), ... * 8],
            ... * 300
        ]

    Term       | What it represents in our model
    -----------+--------------------------------
    Chromosome | The complete sequence of joint angles across 300 frames
    Genome     | Same as chromosome since our individual only has one chromosome
    Gene       | One particular joint's angle at one particular frame -- e.g., frame 100; leg 2; joint 1;
    Allele     | The value assigned to that gene -- e.g., 120deg

    This means we can determine the following numpy shape for each item:
        LegArray:        shape = (3,)
        SpiderArray:     shape = (8, 3)
        IndividualArray: shape = (300, 8, 3)
        PopulationArray: shape = (n, 300, 8, 3)

    The function generates a population of n individuals where each joint angle is a random value
    between the MIN_JOINT_ANGLE and MAX_JOINT_ANGLE control variables.
    It then asserts the population matches the expected shape and returns it
    """
    init_population = np.empty((n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS))
    for i in range(n):

        individual = np.empty((NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS))
        for j in range(NUM_FRAMES):
            rand_spider = np.array([
                rng.uniform(MIN_JOINT_ANGLE, MAX_JOINT_ANGLE, NUM_LEG_SEGS)
                for _ in range(NUM_LEGS)
            ])
            individual[j] = rand_spider

        init_population[i] = individual

    assert init_population.shape == (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    return init_population


def main() -> int:
    init_pop = init_population(20)

    return 0

if __name__ == '__main__':
    sys.exit(main())