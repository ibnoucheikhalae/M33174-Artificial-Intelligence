import numpy as np
import sys
import pickle
import random

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from matplotlib.axes import Axes

NUM_FRAMES = 300
NUM_LEGS = 8
NUM_LEG_SEGS = 3


# Joint limits in radians. (hip, thigh, knee)
# DOESN'T limit crossover/mutation
# Used in: (1) fitness function (how far past) (2) Initialisation/Seeding and 
# (3) Comparison of hip and knee error

RIGHT_MIN = [-1.57, -1.57, -2.0]
RIGHT_MAX = [1.57, 1.57, 1.0]

# Variables kept separate for readbility and comparison
LEFT_MIN = [-1.57, -1.57, -2.0]
LEFT_MAX = [1.57, 1.57, 1.0]


# Leg order: (index 0 = L1) L1, L2, L3, L4, R4, R3, R2, R1
MIN_JOINT_ANGLES = np.array([LEFT_MIN] * 4 + [RIGHT_MIN] * 4)
MAX_JOINT_ANGLES = np.array([LEFT_MAX] * 4 + [RIGHT_MAX] * 4)

#### POSE CYCLE (Step 1 of seeding)
# Keyframe poses, only used when scoring the generation and seeding the intiial population
# SPREAD: feet fanned around the body
# CLOSE: feet close together in pairs with the adjacent leg
# Note: hip movement is the "swing" of the leg, and knee movement is the "lift" thing
# Serve as target_pose_penalty within ff (looking at Frames 0, 150, 299)

SPREAD = np.array(
    [
        [-0.48, -1.3056, -0.8156],  # L1
        [-0.20, -1.3056, -0.8156],  # L2
        [0.20, -1.3056, -0.8156],  # L3
        [0.48, -1.3056, -0.8156],  # L4
        [-0.48, -1.3056, -0.8156],  # R4
        [-0.20, -1.3056, -0.8156],  # R3
        [0.20, -1.3056, -0.8156],  # R2
        [0.48, -1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

#!! todo: can we get the CLOSE to be more dramatic?
CLOSE = np.array(
    [
        [0.24, -1.3056, -0.8156],  # L1
        [0.10, -1.3056, -0.8156],  # L2
        [-0.10, -1.3056, -0.8156],  # L3
        [-0.24, -1.3056, -0.8156],  # L4
        [0.24, -1.3056, -0.8156],  # R4
        [0.10, -1.3056, -0.8156],  # R3
        [-0.10, -1.3056, -0.8156],  # R2
        [-0.24, -1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

# Desired cycle: SPREAD 0, CLOSE 150, SPREAD 299

KEYFRAMES = np.array([0, 150, 299])
TARGET_POSES = np.stack([SPREAD, CLOSE, SPREAD])

#### STEP CYCLE (Step 2 of seeding)
# Two teams: A = L1 + L3, R2 + R4 // B = L2 + L4, R1 + R3
# When A's feet are up B's feet are down (each for half a cycle)
_STEP_PHASE_OFFSETS = np.array([0.0, np.pi] * 4)

# Ipsilateral couples (step together)
COUPLES = np.array([[0, 2], [1, 3], [7, 5], [6, 4]])


def ideal_hip_movement() -> NDArray[np.float64]:
    """SPREAD -> CLOSE -> SPREAD. Used to seed generation 0, not to score.

    Fitness only checks the three standing frames (frames 0, 150, 299). This
    function fills every frame between them so a seed starts with an approximate
    hip path; mutation/crossover is not constrained to it. Thigh and knee stay at
    the standing bend, so only the hips actually move. Movement starts and ends at rest.
    """
    phase = np.interp(np.arange(NUM_FRAMES), KEYFRAMES, [0.0, 1.0, 0.0])
    progress = 3 * phase**2 - 2 * phase**3
    return SPREAD + progress[:, None, None] * (CLOSE - SPREAD)


def sinusoidal_seed_gait(rng: np.random.Generator) -> NDArray[np.float64]:
    """A starting point for leg movement between the keyframe poses.

    Opposite legs are antiphase (Team A lifts while Team B stays down -> four up,
    four down) Sine walk doesn't automatically land on a key frame at 0/299. 20 
    frames at the start and end are slowed down to match the keyframes gradually. 
    Amplitudes, cycle count and pair timing are randomised. STARTING POINT, not TARGET.
    """
    gait = ideal_hip_movement().copy()

    # 2 or 3 steps over 300 frames
    cycles = int(rng.integers(2, 4))

    # gen0 gets one timing nudge to ensure both legs in a pair bare a resemblance at the
    # start, again not a constraint on mutation/crossover.

    couple_timing = rng.normal(0.0, 0.15, len(COUPLES))
    hip_amp = rng.uniform(0.12, 0.22, len(COUPLES))
    thigh_amp = rng.uniform(0.35, 0.55, len(COUPLES))
    knee_amp = rng.uniform(0.50, 0.75, len(COUPLES))
    phase_offsets = _STEP_PHASE_OFFSETS.copy()
    amplitudes = np.empty((NUM_LEGS, NUM_LEG_SEGS))
    for couple, timing, hip, thigh, knee in zip(
        COUPLES, couple_timing, hip_amp, thigh_amp, knee_amp
    ):
        phase_offsets[couple] += timing
        amplitudes[couple] = (hip, thigh, knee)

    phase = (
        2 * np.pi * cycles * np.arange(NUM_FRAMES)[:, None] / NUM_FRAMES
        + phase_offsets[None, :]
    )

    # Turn the extra walk off at the two ends so frames 0 and 299 stay standing.
    frames = np.arange(NUM_FRAMES)
    fade_start = np.clip(frames / 20.0, 0.0, 1.0)  # 0 → 1 over the first 20 frames
    fade_end = np.clip((NUM_FRAMES - 1 - frames) / 20.0, 0.0, 1.0)  # 1 → 0 over the last 20
    fade = np.minimum(fade_start, fade_end)
    fade = (fade * fade * (3.0 - 2.0 * fade))[:, None]  # ease, so it doesn't click on/off

    # 0 when the foot should be down (or at frame 0/299). 1 at the top of a step.
    lift = (np.maximum(np.sin(phase), 0.0) ** 2) * fade

    # Hip spread through a full sine: forward during the lift, back during stance.
    gait[:, :, 0] += fade * np.sin(phase) * amplitudes[None, :, 0]

    # Adding to thigh and knee lifts the foot in 3D view
    gait[:, :, 1] += lift * amplitudes[None, :, 1]
    gait[:, :, 2] += lift * amplitudes[None, :, 2]

    gait += rng.normal(0.0, 0.005, gait.shape)
    return gait


def init_population(
    n: int, seed: Optional[int] = None, seeded_fraction: float = 0.5
) -> NDArray[np.float64]:
    """
    Initialise a population of n individuals
    The structure of each individual is described below:

    - 1 joint angle is a number - e.g., -0.82
    - 1 leg is 3 joint angles (hip, thigh, knee) - e.g., (-0.48, -1.31, -0.82)
    - 1 spider has 8 legs - e.g., ((-0.48, -1.31, -0.82), * 8)
    - 1 individual is the spider's movement over 300 frames:
        [
            [(-0.48, -1.31, -0.82), ... * 8],
            ... * 300
        ]

    Term       | What it represents in our model
    -----------+--------------------------------
    Chromosome | The complete sequence of joint angles across 300 frames
    Genome     | Same as chromosome since our individual only has one chromosome
    Gene       | One particular joint's angle at one particular frame -- e.g., frame 100; leg 2; joint 1;
    Allele     | The value assigned to that gene -- e.g., -0.82 rad

    This means we can determine the following numpy shape for each item:
        LegArray:        shape = (3,)
        SpiderArray:     shape = (8, 3)
        IndividualArray: shape = (300, 8, 3)
        PopulationArray: shape = (n, 300, 8, 3)

    seeded_fraction of the population starts from sinusoidal seed
    gaits (starting points). The rest is all random within the joint 
    limits for diversity. Nothing is pinned or clamped afterwards;
    all 300 frames of every individual are open to mutation/crossover.
    """
    if not 0 <= seeded_fraction <= 1:
        raise ValueError("seeded_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    population = rng.uniform(MIN_JOINT_ANGLES, MAX_JOINT_ANGLES, size=shape)
    for i in range(round(n * seeded_fraction)):
        population[i] = sinusoidal_seed_gait(rng)
    return population


def evaluate_fitness(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Our fitness function. Evaluates the fitness of each individual in a population.
    Takes in a population and returns an array consisting of one fitness value per individual.
    """
    # Defer this import because fitness_function imports the joint constants from GA.
    from GA.fitness_function import fitness_function

    fitness = np.array([fitness_function(gait) for gait in pop])

    if len(fitness) != len(pop):
        raise ValueError(
            f"Fitness shape `{fitness.shape}` does not match 1D population shape: `{((pop.shape[0]),)}`"
        )
    return fitness


def roulette_selection(pop: NDArray[np.float64], fitness: NDArray[np.float64], n: int) -> NDArray[np.float64]:
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
    Uniform whole-leg crossover: children inherit complete leg trajectories.

    Each pair of parents is crossed with probability crossover_rate. When a
    pair crosses, each of the 8 legs is independently swapped (p = 0.5)
    between the two children, taking that leg's ENTIRE 300-frame trajectory
    (all 3 joints) from one parent. Treating a whole leg as the chromosome
    block keeps within-leg coordination intact and introduces no
    frame-to-frame discontinuities inside a leg.

    The result is never clamped: any angle combination the operator produces
    goes to the fitness function as-is.

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

    for i in range(0, len(parents) - 1, 2):
        if rng.random() < crossover_rate:
            swap_legs = rng.random(NUM_LEGS) < 0.5
            offspring[i][:, swap_legs] = parents[i + 1][:, swap_legs]
            offspring[i + 1][:, swap_legs] = parents[i][:, swap_legs]

    return offspring


def mutate(
    pop: NDArray[np.float64], mutation_rate: float, mutation_std: float
) -> NDArray[np.float64]:
    """
    Additive Gaussian mutation on independently selected genes.

    Each gene mutates with probability mutation_rate; a mutated gene has a
    Gaussian step (mean 0, std mutation_std radians) ADDED to it. There is no
    smoothing, no averaging with neighbouring frames, no pinned frames and no
    clamping: a mutation may push an angle outside its joint limits or make
    the motion jerky, and the fitness function's penalties decide whether
    that candidate survives.

    Args:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The population to mutate
        mutation_rate: float
            - Probability of mutating each gene (between 0 and 1)
        mutation_std: float
            - Standard deviation of the Gaussian step, in radians

    Returns:
        mutated: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The mutated population
    """
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not np.isfinite(mutation_std) or mutation_std < 0:
        raise ValueError("mutation_std must be finite and non-negative")

    rng = np.random.default_rng()

    # One random number per gene; True means that gene mutates.
    mask = rng.random(pop.shape) < mutation_rate
    # One Gaussian step per gene, centred on 0 so steps go either way.
    steps = rng.normal(0.0, mutation_std, size=pop.shape)

    return pop + np.where(mask, steps, 0.0)


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
    seeded_fraction: float = 0.5,
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

    seeded_fraction sets how much of the initial population comes from
    sinusoidal seed gaits (the rest is uniform random).

    Returns:
        generations: NDArray[np.float64]; shape = (num_generations+1, population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - Every generation of this training run in order
    """

    # 1) Initialise Population
    pop = init_population(population_size, pop_seed, seeded_fraction)

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


def plot_fitness_history(
    generations: NDArray[np.float64], configuration: str
) -> "Axes":
    """Plot mean fitness and the worst-to-best range for each generation.

    Pass the full history returned by run_GA_training, including generation 0,
    and a readable configuration string for the title, for example:
    "Tournament | population=100 | crossover=0.2 | mutation=0.01 | std=0.1".

    The shaded band is the population's minimum-to-maximum fitness range,
    not a confidence interval. Fitness is recomputed using the current scoring
    function. Saved summaries alone do not contain the worst fitness values.
    Returns the Matplotlib axes so callers can save the figure or call plt.show().
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if (
        generations.ndim != 5
        or generations.shape[2:] != (NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
        or 0 in generations.shape[:2]
    ):
        raise ValueError("Expected a non-empty history of populations from run_GA_training")

    fitness = np.array([evaluate_fitness(population) for population in generations])
    generation_numbers = np.arange(len(generations))
    _, axes = plt.subplots(layout="constrained")
    axes.plot(generation_numbers, fitness.mean(axis=1), label="Mean fitness")
    axes.fill_between(
        generation_numbers, fitness.min(axis=1), fitness.max(axis=1),
        alpha=0.2, label="Worst-best fitness",
    )
    axes.set_title(configuration, wrap=True)
    axes.set_xlabel("Generation (0 = initial population)")
    axes.set_xlim(0, max(1, len(generations) - 1))
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_ylabel("Fitness (higher is better)")
    axes.grid(alpha=0.3)
    axes.legend()
    return axes


def main(printy: bool = False) -> int:
    import matplotlib.pyplot as plt

    population_size = 60
    num_generations = 150
    crossover_rate = 0.6
    mutation_rate = 0.002
    mutation_std = 0.01
    selection_method = tournament_selection
    seeded_fraction = 0.5
    generations = run_GA_training(
        population_size=population_size,
        num_generations=num_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_std=mutation_std,
        selection_method=selection_method,
        seeded_fraction=seeded_fraction,
    )

    fitness_history = np.array(
        [evaluate_fitness(generation) for generation in generations]
    )
    if printy:
        for i, generation_fitness in enumerate(fitness_history):
            print(
                f"Gen {i}: Best fitness: {generation_fitness.max():.4f}; "
                f"average_fitness: {generation_fitness.mean():.4f};"
            )

    last_generation = generations[-1]
    best_index = int(fitness_history[-1].argmax())
    best_gait = last_generation[best_index]

    # Lecturer's check: every joint should visibly evolve. Report how much
    # each joint moves over the best gait so a stuck joint is easy to spot.
    if printy:
        joint_motion = best_gait.std(axis=0)  # (NUM_LEGS, NUM_LEG_SEGS)
        print("Per-joint motion (std over frames, radians):")
        print("leg    hip    thigh  knee")
        leg_names = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
        for name, (hip, thigh, knee) in zip(leg_names, joint_motion):
            print(f"{name}   {hip:6.3f} {thigh:6.3f} {knee:6.3f}")
        from GA.fitness_function import fitness_breakdown, foot_heights, STANCE_Z

        clearance = (foot_heights(best_gait) - STANCE_Z).max(axis=0)
        print("Peak foot clearance (renderer units):", np.round(clearance, 3))
        print("Fitness breakdown:", {k: round(v, 3) for k, v in fitness_breakdown(best_gait).items()})

    summ = {
        "best_fitness": float(fitness_history[-1].max()),
        "average_fitness": float(fitness_history[-1].mean()),
        "best_gait": best_gait,
        "fitness_history": fitness_history,
    }
    with open("recent_run.pkl", "wb") as file:
        pickle.dump(summ, file, protocol=pickle.HIGHEST_PROTOCOL)

    selection_name = selection_method.__name__.removesuffix("_selection").replace("_", " ").title()
    plot_fitness_history(
        generations,
        f"{selection_name} | population={population_size} | generations={num_generations}\n"
        f"crossover={crossover_rate:g} | mutation={mutation_rate:g} | std={mutation_std:g}"
        f" | seeded_fraction={seeded_fraction}",
    )
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(True))
