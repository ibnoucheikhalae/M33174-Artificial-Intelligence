import numpy as np
import sys
import pickle
import random

from numpy.typing import NDArray
from typing import Callable, Optional

NUM_FRAMES = 300
NUM_LEGS = 8
NUM_LEG_SEGS = 3

RIGHT_MIN = [-1.57, -1.57, -2.0]
RIGHT_MAX = [1.57, 1.57, 1.0]

LEFT_MIN = [-1.57, -1.57, -2.0]
LEFT_MAX = [1.57, 1.57, 1.0]

# L1, L2, L3, L4, R4, R3, R2, R1 : Based on the controller
MIN_JOINT_ANGLES = np.array([LEFT_MIN] * 4 + [RIGHT_MIN] * 4)
MAX_JOINT_ANGLES = np.array([LEFT_MAX] * 4 + [RIGHT_MAX] * 4)

# Radians, taken from the smallest scalar values in angle min/max's -- this might be wrong
MUTATION_RANGE_STEP_MIN = 1
MUTATION_RANGE_STEP_MAX = 2

# Start / end pose in the supplied renderer's convention: mirrored legs have
# opposite hip yaw, but equal thigh and knee pitch.
SPREAD = np.array(
    [
        [-0.48, -1.3056, 0.8156],  # L1
        [-0.20, -1.3056, 0.8156],  # L2
        [0.20, -1.3056, 0.8156],  # L3
        [0.48, -1.3056, 0.8156],  # L4
        [-0.48, -1.3056, 0.8156],  # R4
        [-0.20, -1.3056, 0.8156],  # R3
        [0.20, -1.3056, 0.8156],  # R2
        [0.48, -1.3056, 0.8156],  # R1
    ],
    dtype=float,
)


#! Assuming float64, we may be able to optimise this
def init_population(
    n: int, seed: Optional[int] = None, initial_std: Optional[float] = None
) -> NDArray[np.float64]:
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

    Add independent Gaussian noise to the reference, then enforce joint limits.
    initial_std is in radians; None uses uniform random angles for comparison.
    Only the required start/end poses remain fixed during evolution.
    """
    rng = np.random.default_rng(seed)
    shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    if initial_std is None:
        population = rng.uniform(MIN_JOINT_ANGLES, MAX_JOINT_ANGLES, size=shape)
    else:
        if not np.isfinite(initial_std) or initial_std < 0:
            raise ValueError("initial_std must be finite and non-negative")
        # Defer the import: fitness_function imports the joint constants from GA.
        from fitness_function import INTERPOLATED_REFERNCE_GAIT

        population = INTERPOLATED_REFERNCE_GAIT + rng.normal(0.0, initial_std, size=shape)
        population = np.clip(population, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)
    population[:, [0, -1]] = SPREAD
    return population


def evaluate_fitness(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Our fitness function. Evaluates the fitness of each individual in a population.
    Takes in a population and returns an array consisting of one fitness value per individual.
    """
    # Defer this import because fitness_function imports the joint constants from GA.
    from fitness_function import fitness_function

    fitness = np.array([fitness_function(gait) for gait in pop])

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

    # Penalty-based fitness can be negative. Shifting preserves the ordering;
    # epsilon also gives equal-fitness populations a valid uniform distribution.
    weights = fitness - min(0.0, float(fitness.min())) + np.finfo(float).eps
    normalised = weights / weights.sum()
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


def tournament_selection(
    pop: NDArray[np.float64], fitness: NDArray[np.float64], n: int, tournament_size: int = 2
) -> NDArray[np.float64]:
    """
    Uses tournament selection to select which individuals of the population will proceed to
    crossover.

    Args:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The population to select from
        fitness: NDArray[np.float64]; shape = (population_size)
            - The fitness values of the individuals within that population
        n: int;
            - The number of individuals selected
        tournament_size: int;
            - Candidates sampled per tournament; larger values increase selection pressure

    Returns:
        selected_parents: NDArray[np.float64]; shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    """
    if tournament_size < 1:
        raise ValueError("tournament_size must be positive")
    selected_parents = np.empty(
        (n,) + pop.shape[1:], dtype=np.float64
    )  # (n,) is python 1-tuple weirdness
    for i in range(n):
        candidates = random.choices(range(len(fitness)), k=tournament_size)
        winner = max(candidates, key=lambda index: fitness[index])
        selected_parents[i] = pop[winner]

    if selected_parents.shape != (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS):
        raise ValueError(
            f"Selected Parents wrong shape! Expected: {(n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)}; Actual: {selected_parents.shape};"
        )
    return selected_parents


def crossover(
    parents: NDArray[np.float64], crossover_rate: float
) -> NDArray[np.float64]:
    """
    Probabilistically combine each pair using a random weighted average.

    Args:
        parents: Population with shape (n, frames, legs, joints).
        crossover_rate: Probability of crossing each parent pair, between 0 and 1.

    Returns:
        Offspring with the same shape as parents. If the population size is
        odd, the final individual is copied unchanged.
    """
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1")

    rng = np.random.default_rng()
    offspring = parents.copy()

    # Arithmetic crossover is a standard real-valued GA operator (MATLAB):
    # https://www.mathworks.com/help/gads/genetic-algorithm-options.html
    # Its weighted averages combine parents without adding frame-to-frame
    # switches, helping preserve movement continuity.
    for i in range(0, len(parents) - 1, 2):
        if rng.random() < crossover_rate:
            # One weight for the whole gait avoids switching parents between
            # adjacent frames. Both children lie between their parents.
            weight = rng.random()
            difference = parents[i + 1] - parents[i]
            offspring[i] = parents[i] + weight * difference
            offspring[i + 1] = parents[i + 1] - weight * difference

    # Guard against floating-point roundoff at a joint limit.
    return np.clip(offspring, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)


def mutate(
    pop: NDArray[np.float64], mutation_rate: float, mutation_std: float
) -> NDArray[np.float64]:
    """
    Mutate selected angles around their temporal neighbours' mean.

    This biases mutation towards local continuity, using only the existing
    gait. Gaussian variation and fitness selection determine the movement.

    Args:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The population to mutate
        mutation_rate: float
            - Probability of mutating each interior gene (between 0 and 1)
        mutation_std: float
            - Standard deviation around the neighbours' mean, in radians.
              Zero still averages selected angles; it only removes random variation.

    Returns:
        mutated: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The mutated population
    """

    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not np.isfinite(mutation_std) or mutation_std < 0:
        raise ValueError("mutation_std must be finite and non-negative")

    rng = np.random.default_rng()

    # Generate one random number per gene.
    # Comparing against mutation rate gives a boolean mask: True means that gene mutates
    mask = rng.random(pop.shape) < mutation_rate
    # Boundary poses are fixed requirements, not genes to optimise.
    mask[:, [0, -1]] = False

    # Generate one Gaussian step per gene (normal), centered on 0 to allow for positive or negative
    # mutation_std controls the random variation around the neighbours' mean.
    steps = rng.normal(0.0, mutation_std, size=pop.shape)

    # Nearby frames should usually have similar angles. Centre Gaussian
    # mutation on their mean to favour continuity while retaining exploration.
    # Read the original gait so updates are simultaneous; no reference is used.
    neighbour_mean = 0.5 * (pop[:, :-2] + pop[:, 2:])
    mutated = pop.copy()
    mutated[:, 1:-1] = np.where(
        mask[:, 1:-1], neighbour_mean + steps[:, 1:-1], pop[:, 1:-1]
    )

    # Clamp each angle to its joints valid range
    return np.clip(mutated, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)


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
    initial_std: Optional[float] = None,
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

    initial_std sets the initial Gaussian noise in radians; None starts uniformly.

    Returns:
        generations: NDArray[np.float64]; shape = (num_generations+1, population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - Every generation of this training run in order
    """

    # 1) Initialise Population
    pop = init_population(population_size, pop_seed, initial_std)

    # Extra step: Store all generations so we can do data analysis on them later
    # num_generations+1 as initial population is considered generation 0
    generations = np.empty((num_generations + 1,) + pop.shape)
    generations[0] = pop

    # 2) Training Loop
    for gen in range(
        1, num_generations + 1
    ):  # +1 because the initial population is gen 0
        # 2.a) Evaluate Population Fitness
        population_fitness = evaluate_fitness(pop)

        # 2.b) Select Individuals for Crossover
        # n=population_size means population won't shrink/grow
        selected_parents = selection_method(pop, population_fitness, population_size)

        # 2.c) Apply Crossover
        offspring = crossover(selected_parents, crossover_rate)

        # 2.d) Mutate new population
        new_pop = mutate(offspring, mutation_rate, mutation_std)

        # Preserve the best discovered gait so crossover/mutation cannot lose it. A single elite.
        new_pop[0] = pop[int(population_fitness.argmax())]

        # 2.e)
        pop = new_pop
        generations[gen] = pop

    return generations


def main(printy: bool = False) -> int:
    #! Note: Temporary stand-in values for now
    generations = run_GA_training(
        population_size=1,
        num_generations=1,
        crossover_rate=0.2,
        mutation_rate=0.01,
        mutation_std=0.01,
        selection_method=tournament_selection,
        initial_std=0
    )
    if printy:
        for i, gen in enumerate(generations):
            final_fitness = evaluate_fitness(gen)
            best, average = np.max(final_fitness), np.mean(final_fitness)
            print(f"Gen {i}: Best fitness: {best}; average_fitness: {average};")

    last_generation = generations[-1]
    fitneses = evaluate_fitness(last_generation)
    best_fitness = np.max(fitneses)
    best_indx = np.where(fitneses == best_fitness)
    summ = {
        "best_fitness": best_fitness,
        "average_fitenss": np.mean(last_generation),
        "best_gait": last_generation[best_indx],
        "best_generation": last_generation
    }
    with open("recent_run.pkl", "wb") as file:
        pickle.dump(summ, file, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == "__main__":
    sys.exit(main(True))
