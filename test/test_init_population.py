import numpy as np
import pytest

from GA import (
    MAX_JOINT_ANGLES,
    MIN_JOINT_ANGLES,
    NUM_FRAMES,
    NUM_LEG_SEGS,
    NUM_LEGS,
    init_population,
)


@pytest.mark.parametrize("n", [1, 5, 50])
def test_shape(n: int) -> None:
    population = init_population(n)
    assert population.shape == (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)


def test_values_within_joint_limits():
    population = init_population(20)
    assert (population >= MIN_JOINT_ANGLES).all()
    assert (population <= MAX_JOINT_ANGLES).all()


def test_noinvalid():
    population = init_population(10)
    assert np.isfinite(population).all()
