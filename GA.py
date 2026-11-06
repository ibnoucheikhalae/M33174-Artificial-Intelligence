import numpy as np
import sys
import random

from numpy.typing import NDArray
from typing import Callable, Optional

NUM_FRAMES = 300
NUM_LEGS = 8
NUM_LEG_SEGS = 3

RIGHT_MIN = [-1.57, -1.57, -1.0]
RIGHT_MAX = [1.57, 1.57, 1.0]

LEFT_MIN = [-1.57, -1.57, -1.0]
LEFT_MAX = [1.57, 1.57, 1.0]

# L1, L2, L3, L4, R4, R3, R2, R1 : Based on the controller
MIN_JOINT_ANGLES = np.array([LEFT_MIN] * 4 + [RIGHT_MIN] * 4)
MAX_JOINT_ANGLES = np.array([LEFT_MAX] * 4 + [RIGHT_MAX] * 4)


#! Assuming float64, we may be able to optimise this
def init_population(n: int, seed: Optional[int] = None) -> NDArray[np.float64]:
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
    rng = np.random.default_rng(seed)
    init_population = rng.uniform(
        MIN_JOINT_ANGLES, MAX_JOINT_ANGLES, size=(n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    )

    return init_population


def evaluate_fitness(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    from fitness_function import fitness_function_batch

    fitness = fitness_function_batch(pop)

    if len(fitness) != len(pop):
        raise ValueError(
            f"Fitness shape `{fitness.shape}` does not match 1D population shape: `{((pop.shape[0]),)}`"
        )
    return fitness


def roulette_selection(
    pop: NDArray[np.float64], fitness: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """
    Uses roulette selection to select which individuals of the population will proceed to
    crossover.

    Args:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The population to select from
        fitness: NDArray[np.float64]; shape = (population_size)
            - The fitness values of the individuals within that population
        n: int;
            - The number of individuals selected

    Returns:
        selected_parents: NDArray[np.float64]; shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    """
    selected_parents = np.empty((n,) + pop.shape[1:], dtype=np.float64)

    normalised = fitness / fitness.sum()
    cumulative_fitness = np.cumsum(normalised)
    cumulative_fitness[-1] = 1.0  # Force last value to 1 to avoid 0.999999...

    for i in range(n):
        rand_n = random.uniform(0, 1)
        individual_n = 0
        for fitvalue in cumulative_fitness:
            if rand_n <= fitvalue:
                selected_parents[i] = pop[individual_n]
                break
            individual_n += 1

    if selected_parents.shape != (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS):
        raise ValueError(
            f"Selected Parents wrong shape! Expected: {(n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)}; Actual: {selected_parents.shape};"
        )
    return selected_parents


TOURNAMENT_SIZE = 3
ELITE_FRACTION = 0.05


def tournament_selection(
    pop: NDArray[np.float64],
    fitness: NDArray[np.float64],
    n: int,
    tournament_size: int = TOURNAMENT_SIZE,
) -> NDArray[np.float64]:
    """
    Tournament selection: for each parent, pick tournament_size random individuals
    and keep the fittest.

    Returns: selected parents, shape (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    """
    rng = np.random.default_rng()
    contenders = rng.integers(0, len(fitness), size=(n, tournament_size))
    winners = contenders[np.arange(n), np.argmax(fitness[contenders], axis=1)]
    return pop[winners].copy()


def crossover(
    parents: NDArray[np.float64],
    crossover_rate: float,
    blend_frames: int = 10,
) -> NDArray[np.float64]:
    """
    Two-point time-block crossover with soft seams.

    Each consecutive pair of parents is crossed with probability
    `crossover_rate`. If crossed, two cut points are chosen along the frame
    axis. Inside the block the children swap parents; the edges of the block
    are cross faded over ~`blend_frames` frames so there is no sudden jump.

    Because each child frame is a weighted average of two in-range angles,
    children stay inside the joint limits.
    """
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1")

    rng = np.random.default_rng()
    offspring = parents.copy()
    kernel = np.ones(blend_frames) / blend_frames

    for i in range(0, len(parents) - 1, 2):
        if rng.random() >= crossover_rate:
            continue  # this pair passes through unchanged

        a, b = np.sort(rng.integers(0, NUM_FRAMES + 1, size=2))
        if a == b:
            continue

        # 1 inside the block, 0 outside, then softened at the edges
        w = np.zeros(NUM_FRAMES)
        w[a:b] = 1.0
        if blend_frames > 1:
            padded = np.pad(w, blend_frames, mode="edge")
            w = np.convolve(padded, kernel, mode="same")[blend_frames:-blend_frames]
        w = w[:, None, None]  # broadcast over legs and joints

        offspring[i] = (1 - w) * parents[i] + w * parents[i + 1]
        offspring[i + 1] = w * parents[i] + (1 - w) * parents[i + 1]

    return offspring


# These set how wide a bump can be, in frames. Each bump gets a random width between 10 and 60.
BUMP_MIN_WIDTH = 10
BUMP_MAX_WIDTH = 60


def mutate(
    pop: NDArray[np.float64], mutation_rate: float, mutation_std: float
) -> NDArray[np.float64]:
    """
    Smooth mutation. Each joint of each leg of each individual has a
    mutation_rate chance of getting a bump: a Gaussian-shaped offset added over a
    window of frames, so the motion changes gently instead of jumping.

    Args:
        mutation_rate: chance (0 to 1) that each leg-joint gets a bump
        mutation_std: standard deviation of the bump height in radians
    """
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not np.isfinite(mutation_std) or mutation_std < 0:
        raise ValueError("mutation_std must be finite and non-negative")

    rng = np.random.default_rng()
    mutated = pop.copy()
    frames = np.arange(NUM_FRAMES)

    hits = np.argwhere(rng.random((len(pop), NUM_LEGS, NUM_LEG_SEGS)) < mutation_rate)
    for individual, leg, joint in hits:
        centre = rng.integers(NUM_FRAMES)
        width = rng.integers(BUMP_MIN_WIDTH, BUMP_MAX_WIDTH + 1)
        height = rng.normal(0.0, mutation_std)
        mutated[individual, :, leg, joint] += height * np.exp(
            -0.5 * ((frames - centre) / width) ** 2
        )

    return np.clip(mutated, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)


def elitism(
    pop: NDArray[np.float64],
    fitness: NDArray[np.float64],
    elite_fraction: float = ELITE_FRACTION,
) -> NDArray[np.float64]:

    elite_count = max(1, int(len(pop) * elite_fraction))

    elite_indices = np.argsort(fitness)[-elite_count:]

    return pop[elite_indices].copy()


def run_GA_training(
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
    selection_method: Callable[
        [NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]
    ],
    pop_seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Runs the actual genetic algorithim training using the input control variables.

    Follows the general GA Flow:
    1) Initialise Population
    2) Training Loop (until terminating condition)
        2.a) Evaluate Population Fitness
        2.b) Select the Individuals for Crossover
        2.c) Apply Crossover -- generate new population
        2.d) Mutate new population
        2.e) pop = new_pop
    3) Return all generations for exterior data analysis

    Returns:
        generations: NDArray[np.float64]; shape = (num_generations+1, population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - Every generation of this training run in order
    """

    # 1) Initialise Population
    pop = init_population(population_size, pop_seed)

    # Extra step: Store all generations so we can do data analysis on them later
    # num_generations+1 as initial population is considered generation 0
    generations = np.empty((num_generations + 1,) + pop.shape)
    generations[0] = pop

    # 2) Training Loop
    for gen in range(
        1, num_generations + 1
    ):  # +1 because the initial population is gen 0
        # Evaluate Population Fitness
        population_fitness = evaluate_fitness(pop)

        elites = elitism(pop, population_fitness)

        # Select Individuals for Crossover
        selected_parents = selection_method(
            pop, population_fitness, population_size - len(elites)
        )

        # 2.c) Apply Crossover
        offspring = crossover(selected_parents, crossover_rate)

        # 2.d) Mutate new population
        new_pop = mutate(offspring, mutation_rate, mutation_std)

        # 2.e)
        pop = np.concatenate((elites, new_pop), axis=0)
        generations[gen] = pop

    return generations


def main(printy: bool = False) -> int:
    generations = run_GA_training(
        population_size=100,
        num_generations=50,
        crossover_rate=0.6,
        mutation_rate=0.05,
        mutation_std=0.3,
        selection_method=tournament_selection,
    )

    if printy:
        for i, gen in enumerate(generations):
            final_fitness = evaluate_fitness(gen)
            best, average = np.max(final_fitness), np.mean(final_fitness)
            print(f"Gen {i}: Best fitness: {best}; average_fitness: {average};")

    return 0


if __name__ == "__main__":
    sys.exit(main(True))
