"""Evolve spider gaits and automatically export qualifying individuals for the NN.

The script's main function writes one archive under GAIT_DATA_DIRECTORY when an individual's
fitness exceeds FITNESS_SAVE_THRESHOLD. All generations, including the initial
and final populations, are considered. Gaits stay in radians and are grouped by
run so descendants can remain together in the NN's training/test split.
"""

import hashlib
import pickle
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from GA.fitness_function import MAX_FITNESS, fitness_function
from GA.gait_constants import (
    CLOSE,
    KEYFRAMES,
    NUM_FRAMES,
    SPREAD,
    _STEP_PHASE_OFFSETS,
)
from GA.kinematics import NUM_LEG_SEGS, NUM_LEGS, SOCKET_HEADINGS

# Fitness is at most MAX_FITNESS, 0.9 saves only those who get 90% of MAX_FITNESS;
# save distinct individuals strictly above this threshold.
FITNESS_SAVE_THRESHOLD = MAX_FITNESS * 0.762
GAIT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "NN" / "data"
NUM_STEPS = 1  # Complete spread-close-spread cycles and knee lifts in a seed gait.


def ideal_hip_movement() -> NDArray[np.float64]:
    """SPREAD -> CLOSE -> SPREAD. Used to seed generation 0, not to score.

    Fitness only checks the three standing frames (frames 0, 150, 299). This
    function fills every frame between them so a seed starts with an approximate
    hip path; mutation/crossover is not constrained to it. Thigh and knee stay at
    the standing bend, so only the hips actually move. Movement starts and ends at rest.
    """
    cycle = NUM_STEPS * np.interp(np.arange(NUM_FRAMES), KEYFRAMES, [0.0, 0.5, 1.0])
    phase = np.interp(cycle % 1.0, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
    progress = 3 * phase**2 - 2 * phase**3
    return SPREAD + progress[:, None, None] * (CLOSE - SPREAD)


def sinusoidal_seed_gait() -> NDArray[np.float64]:
    """A starting point for leg movement between the keyframe poses.

    Opposite legs are antiphase (Team A lifts while Team B stays down -> four up,
    four down). NUM_STEPS controls both hip sweeps and knee lifts.
    STARTING POINT, not TARGET.
    """
    gait = ideal_hip_movement().copy()
    cycle = NUM_STEPS * np.interp(np.arange(NUM_FRAMES), KEYFRAMES, [0.0, 0.5, 1.0])
    phase = 2 * np.pi * cycle[:, None] + _STEP_PHASE_OFFSETS[None, :]

    # 0 when the foot should be down; 1 at the top of a step.
    lift = np.maximum(np.sin(phase), 0.0) ** 2

    # Sweep forward while lifted and backward while planted; mirror right-leg yaw.
    hip = (SPREAD[:, 0] + CLOSE[:, 0]) / 2
    hip = hip + 0.25 * np.cos(phase) * np.sign(SOCKET_HEADINGS)
    gait[:, :, 0] = hip

    # Adding to thigh and knee lifts the foot in 3D view.
    gait[:, :, 1] += lift * 0.45
    gait[:, :, 2] += lift * 0.625

    return gait


def init_population(
    n: int, seed: Optional[int] = None, initial_std: float = 0.005
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

    Every individual starts from the same hip-and-knee reference gait plus
    independent Gaussian noise with initial_std radians of standard deviation.
    Nothing is pinned or clamped afterwards;
    all 300 frames of every individual are open to mutation/crossover.
    """
    if not np.isfinite(initial_std) or initial_std < 0:
        raise ValueError("initial_std must be finite and non-negative")

    rng = np.random.default_rng(seed)
    shape = (n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    return sinusoidal_seed_gait() + rng.normal(0.0, initial_std, size=shape)


def evaluate_fitness(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Our fitness function. Evaluates the fitness of each individual in a population.
    Takes in a population and returns an array consisting of one fitness value per individual.
    """
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


def save_training_gaits(
    gaits: NDArray[np.float64],
    fitness: NDArray[np.float64],
    generation_indices: NDArray[np.int64],
    directory: Path = GAIT_DATA_DIRECTORY,
) -> Path | None:
    """Save qualifying gaits, scores, and source generations collected by a callback.

    Gaits have shape (N, 300, 8, 3); fitness and generation_indices have shape
    (N,). Exact duplicates are stored once, using their first occurrence.
    """
    if len(gaits) == 0:
        return None

    # unique compares whole gaits (axis 0); indices identify their first occurrence.
    # Sorting those indices restores the original generation/population order.
    _, first_occurrences = np.unique(gaits, axis=0, return_index=True)
    first_occurrences.sort()
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"run_{uuid4().hex}.npz"
    np.savez_compressed(
        output_path,
        gaits=gaits[first_occurrences],
        fitness=fitness[first_occurrences],
        generation_indices=generation_indices[first_occurrences],
        threshold=FITNESS_SAVE_THRESHOLD,
    )
    return output_path


def run_GA(
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
    selection_method: Callable[
        [NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]
    ],
    pop_seed: Optional[int] = None,
    initial_std: float = 0.005,
    on_generation: Callable[[int, NDArray[np.float64], NDArray[np.float64]], None] | None = None,
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
    3) Return the final population

    ``initial_std`` sets the initial Gaussian noise in radians around the
    common hip-and-knee reference gait.

    ``on_generation`` is an optional observer called with the generation index and
    its fitness vector and population, including generation zero and the final generation.
    The observer must not modify either array and must copy any gaits it retains.
    The algorithm itself does not log progress, filter training data, or write files;
    callers own these responsibilities.

    Returns:
        pop: NDArray[np.float64]; shape = (population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
            - The final population; previous populations are discarded
    """

    # 1) Initialise Population
    pop = init_population(population_size, pop_seed, initial_std)

    # 2) Training Loop
    for gen in range(1, num_generations + 1):  # +1 because the initial population is gen 0
        # 2.a) Evaluate Population Fitness
        population_fitness = evaluate_fitness(pop)
        if on_generation is not None:
            on_generation(gen - 1, population_fitness, pop)

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

    if on_generation is not None:
        # The final population has not yet been scored by the evolution loop.
        on_generation(num_generations, evaluate_fitness(pop), pop)
    return pop


def plot_fitness_history(fitness: NDArray[np.float64], configuration: str) -> "Axes":
    """Plot mean and best fitness with the worst-to-best range per generation.

    Pass recorded (mean, worst, best) fitness per generation, including generation 0,
    and a readable configuration string for the title, for example:
    "Tournament | population=100 | crossover=0.2 | mutation=0.01 | std=0.1".

    The shaded band is the population's minimum-to-maximum fitness range,
    not a confidence interval. Best fitness is drawn explicitly so DeltaF
    (final best minus generation-0 best) is readable from the figure.
    Fitness is recorded by the generation callback.
    Returns the Matplotlib axes so callers can save the figure or call plt.show().
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    generation_numbers = np.arange(len(fitness))
    mean_fitness = fitness[:, 0]
    worst_fitness = fitness[:, 1]
    best_fitness = fitness[:, 2]
    f0 = float(best_fitness[0])
    f_star = float(best_fitness.max())
    delta_f = f_star - f0
    g_star = int(best_fitness.argmax())

    _, axes = plt.subplots(layout="constrained")
    axes.fill_between(
        generation_numbers,
        worst_fitness,
        best_fitness,
        alpha=0.2,
        label="Worst–best range",
    )
    axes.plot(generation_numbers, mean_fitness, label="Mean fitness", color="C0")
    axes.plot(
        generation_numbers,
        best_fitness,
        label="Best fitness",
        color="C1",
        linewidth=2.0,
    )
    axes.scatter(
        [0, g_star],
        [f0, f_star],
        color="C1",
        zorder=5,
        label=f"ΔF = {delta_f:.4f}  (F0={f0:.4f} → F*={f_star:.4f})",
    )
    axes.set_title(configuration, wrap=True)
    axes.set_xlabel("Generation (0 = initial population)")
    axes.set_xlim(0, max(1, len(fitness) - 1))
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_ylabel("Fitness (higher is better)")
    axes.grid(alpha=0.3)
    axes.legend()
    return axes


def main(printy: bool = False) -> int:
    """Run the configured GA, export NN data, save the best gait, and plot fitness."""
    import matplotlib.pyplot as plt

    # Tuned for DeltaF / final fitness (short grid + multi-seed checks).
    # Old pm=0.05 + initial_std=0.1 stalled or stayed deeply negative.
    # Best band: low mutation, high crossover, stronger tournament, tight seed.
    # seed=42, pop=100, gens=300 -> F0~1.37 F*~1.45 dF~0.08 (still climbing).
    population_size = 100
    num_generations = 500
    crossover_rate = 0.9
    mutation_rate = 0.002
    mutation_std = 0.005
    tournament_size = 7
    selection_method = lambda pop, fitness, n: tournament_selection(
        pop, fitness, n, tournament_size=tournament_size
    )
    initial_std = 0.005
    data_directory = Path(GAIT_DATA_DIRECTORY).resolve()
    print(
        f"NN data export: fitness > {FITNESS_SAVE_THRESHOLD}; directory={data_directory}. "
        "Archives are saved after all generations finish.",
        flush=True,
    )
    fitness_history = []
    qualifying_gaits = []
    qualifying_fitness = []
    generation_indices = []
    fitneses = np.empty(0)

    def record_generation(
        generation: int, fitness: NDArray[np.float64], population: NDArray[np.float64]
    ) -> None:
        """Collect summary scores and qualifying gaits, and optionally display progress."""
        nonlocal fitneses
        fitneses = fitness
        fitness_history.append((fitness.mean(), fitness.min(), fitness.max()))
        for index in np.flatnonzero(fitness > FITNESS_SAVE_THRESHOLD):
            gait = population[index]
            qualifying_gaits.append(gait.copy())
            qualifying_fitness.append(fitness[index])
            generation_indices.append(generation)
        if printy:
            print(
                f"Generation {generation}/{num_generations}: "
                f"best={fitness.max():.6f}; mean={fitness.mean():.6f}; "
                f"worst={fitness.min():.6f}; "
                f"qualifying={np.count_nonzero(fitness > FITNESS_SAVE_THRESHOLD)}",
                flush=True,
            )

    last_generation = run_GA(
        population_size=population_size,
        num_generations=num_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_std=mutation_std,
        selection_method=selection_method,
        initial_std=initial_std,
        on_generation=record_generation,
    )
    archive_path = save_training_gaits(
        np.asarray(qualifying_gaits), np.asarray(qualifying_fitness),
        np.asarray(generation_indices, dtype=np.int64), data_directory
    )
    if archive_path is None:
        print(f"No gaits exceeded fitness {FITNESS_SAVE_THRESHOLD}; no NN data saved.", flush=True)
    else:
        print(f"Saved qualifying NN training gaits to {archive_path}", flush=True)

    best_fitness = np.max(fitneses)
    best_indx = np.where(fitneses == best_fitness)
    summ = {
        "best_fitness": best_fitness,
        "average_fitenss": np.mean(last_generation),
        "best_gait": last_generation[best_indx],
        "best_generation": last_generation,
        "fitness_history": np.asarray(fitness_history),
    }
    with open("recent_run.pkl", "wb") as file:
        pickle.dump(summ, file, protocol=pickle.HIGHEST_PROTOCOL)

    history = np.asarray(fitness_history)
    best_curve = history[:, 2]
    f0 = float(best_curve[0])
    f_star = float(best_curve.max())
    delta_f = f_star - f0
    print(
        f"DeltaF: F0(best)={f0:.6f}; F*(best)={f_star:.6f}; "
        f"DeltaF={delta_f:.6f} at generation {int(best_curve.argmax())}",
        flush=True,
    )

    selection_name = f"Tournament (size={tournament_size})"
    plot_fitness_history(
        history,
        f"{selection_name} | population={population_size} | generations={num_generations}\n"
        f"crossover={crossover_rate:g} | mutation={mutation_rate:g} | std={mutation_std:g}"
        f" | initial_std={initial_std}",
    )
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(True))
