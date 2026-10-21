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
    expected_shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    init_population = np.empty(expected_shape, dtype=np.float64)
    for i in range(n):

        individual = np.empty((NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS), dtype=np.float64)
        for j in range(NUM_FRAMES):
            rand_spider = np.array([
                rng.uniform(MIN_JOINT_ANGLE, MAX_JOINT_ANGLE, NUM_LEG_SEGS)
                for _ in range(NUM_LEGS)
            ], dtype=np.float64)
            individual[j] = rand_spider

        init_population[i] = individual

    if init_population.shape != expected_shape:
        raise ValueError(
            f"Population initialised to wrong shape! Expected: {expected_shape}; Actual: {init_population.shape};"
        )
    return init_population

def evaluate_fitness(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Our fitness function. Evaluates the fitness of each individual in a population.
    Takes in a population and returns an array consisting of one fitness value per individual.
    """
    fitness = np.zeros(pop.shape[0])  #! Stub function while I lay out large structure

    if len(fitness) != len(pop):
        raise ValueError(
            f"Fitness shape `{fitness.shape}` does not match 1D population shape: `{(pop.shape[0]),}`"
        )
    return fitness

def selection_method(
        pop: NDArray[np.float64], fitness: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """
    Uses a selection algorithm to select which individuals of the population will proceed to
    crossover.

    ! Note: Does tournament selection as stub for now.

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
    selected_parents = np.empty((n,) + pop.shape[1:], dtype=np.float64)  # (n,) is python 1-tuple weirdness
    for i in range(n):
        idx1 = random.randrange(fitness.shape[0])
        idx2 = random.randrange(fitness.shape[0])

        selected_parents[i] = pop[idx1 if fitness[idx1] > fitness[idx2] else idx2]

    if selected_parents.shape != (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS):
        raise ValueError(
            f"Selected Parents wrong shape! Expected: {(n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)}; Actual: {selected_parents.shape};"
        )
    return selected_parents


def crossover(parents: NDArray[np.float64], crossover_point: int) -> NDArray[np.float64]:
    """
    Apply crossover on the selected parent population, resulting in our new population.

    Args:
        parents: NDArray[np.float64]; shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The individuals selected for crossover by our selection algorithm
        crossover_point: int
            - The index at which the crossover occurs. In the second dimension (where in 300 frames)

    Returns:
        offspring: NDArray[np.float64]; shape = (parents.shape[0], NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The new population generated from applying the crossover to the parents
    """
    if not 0 <= crossover_point <= parents.shape[1]:
        raise ValueError("crossover_point must be between 0 and the number of frames")

    offspring = parents.copy()

    for i in range(0, len(parents) - 1, 2):
        offspring[i, crossover_point:] = parents[i + 1, crossover_point:]
        offspring[i + 1, crossover_point:] = parents[i, crossover_point:]

    if offspring.shape != parents.shape:
        raise ValueError(
            f"Generated Offspring wrong shape! Expected: {parents.shape}; Actual: {offspring.shape};"
        )
    return offspring


def mutate(pop: NDArray[np.float64], mutation_rate: float) -> NDArray[np.float64]:
    """
    Mutate a population. Each individual has a mutation_rate% of being mutated.

    Args:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The population to mutate
        mutation_rate: float
            - The percentage chance per individual for mutation (between 0 and 1)

    Returns:
        mutated: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The mutated population
    """
    mutated = pop
    for individual in mutated:
        if random.random() >= mutation_rate:
            #! Stub method. Do not mutate for now
            pass

    if mutated.shape != pop.shape:
        raise ValueError(
            f"Mutated Population wrong shape! Expected: {pop.shape}; Actual: {mutated.shape};"
        )
    return mutated


def main() -> int:
    """
    Intended Flow:
    1) Initialise Population
    2) Training Loop (until terminating condition)
        2.a) Evaluate Population Fitness
        2.b) Select the Individuals for Crossover
        2.c) Apply Crossover -- generate new population
        2.d) Mutate new population
        2.e) pop = new_pop
    3) Evaluate final population fitness & display results
    """
    # 0) Define control variables
    #! Note: Temporary stand-in values for now
    population_size = 20
    num_generations = 20
    crossover_point = 150
    mutation_rate = 0.01

    # 1) Initialise Population
    pop = init_population(population_size)

    # Extra step: Store all generations so we can do data analysis on them later
    generations = np.empty((num_generations + 1,) + pop.shape)
    generations[0] = pop

    # 2) Training Loop
    for gen in range(1, num_generations + 1):  # +1 because the initial population is gen 0

        # 2.a) Evaluate Population Fitness
        population_fitness = evaluate_fitness(pop)

        # 2.b) Select Individuals for Crossover
        # n=population_size means population won't shrink/grow
        selected_parents = selection_method(pop, population_fitness, n=population_size)

        # 2.c) Apply Crossover
        offspring = crossover(selected_parents, crossover_point)

        # 2.d) Mutate new population
        new_pop = mutate(offspring, mutation_rate)

        # 2.e)
        pop = new_pop
        generations[gen] = pop

    # 3) Evaluate final population fitness & display results
    final_fitness = evaluate_fitness(pop)
    best, average = np.max(final_fitness), np.mean(final_fitness)
    print(f"Final population fitness: {final_fitness}; average: {average}; best: {best}")

    return 0

if __name__ == '__main__':
    sys.exit(main())