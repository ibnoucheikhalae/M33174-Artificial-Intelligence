"""Diagnostic graphs for a `run_GA_training` run.
Usage:
    from GA.GA import run_GA_training, evaluate_fitness, tournament_selection
    from GA.evolution_graphs import (
        collect_generation_summary, plot_fitness_evolution,
        plot_joint_evolution_heatmap, plot_joint_diversity,
        plot_initial_vs_evolved,
    )

    generations = run_GA_training(population_size=60, num_generations=150,
                                   crossover_rate=0.6, mutation_rate=0.002,
                                   mutation_std=0.01,
                                   selection_method=tournament_selection)
    summary = collect_generation_summary(generations, evaluate_fitness)
    out_dir = Path(".")
    plot_fitness_evolution(summary, out_dir)
    plot_joint_evolution_heatmap(summary, out_dir)
    plot_joint_diversity(summary, out_dir)
    plot_initial_vs_evolved(summary, out_dir)

Graphs produced:
    graph1_fitness_evolution.png        best and mean fitness per generation
    graph2_joint_evolution_heatmap.png  per-joint trajectory change per generation
    graph3_joint_diversity.png          population diversity per joint per generation
    graph4_initial_vs_evolved.png       generation-0 best vs best-ever trajectories

`collect_generation_summary` relies on `run_GA_training`'s single-elite policy
(the previous generation's best individual is always carried into the next
one unchanged), which makes the best fitness per generation monotonically
non-decreasing. That means the best individual of the final generation is
also the best-ever individual, so it is used as "evolved" in graph 4.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from GA.kinematics import LEG_LABELS, NUM_LEGS, NUM_LEG_SEGS

JOINT_NAMES = ("hip", "thigh", "knee")
JOINT_LABELS = [f"{leg}_{joint}" for leg in LEG_LABELS for joint in JOINT_NAMES]

FitnessFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def collect_generation_summary(
    generations: NDArray[np.float64], evaluate_fitness: FitnessFunction
) -> dict:
    """Turn the raw output of `run_GA_training` into what the graphs need.

    Args:
        generations: array from `run_GA_training`, shape
            (num_generations+1, population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
        evaluate_fitness: our shared fitness function (higher is better)
    """
    fitness_per_gen = np.array([evaluate_fitness(pop) for pop in generations])  # (gens, pop)
    best_index_per_gen = fitness_per_gen.argmax(axis=1)
    best_gait_per_gen = generations[np.arange(len(generations)), best_index_per_gen]
    return {
        "best_fitness_hist": fitness_per_gen.max(axis=1),
        "mean_fitness_hist": fitness_per_gen.mean(axis=1),
        "best_gait_per_gen": best_gait_per_gen,      # (gens, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
        "population_per_gen": generations,           # (gens, pop, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
        "best_ever_gait": best_gait_per_gen[-1],      # (NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    }


def plot_fitness_evolution(summary: dict, out: Path) -> Path:
    """Graph 1: best and mean fitness against generation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["best_fitness_hist"], label="Best fitness in generation")
    ax.plot(summary["mean_fitness_hist"], label="Mean fitness in generation")
    ax.set(xlabel="Generation", ylabel="Fitness (higher is better)",
           title="Fitness evolution of the GA")
    ax.legend()
    fig.tight_layout()
    path = out / "graph1_fitness_evolution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_joint_evolution_heatmap(summary: dict, out: Path) -> Path:
    """Graph 2: for each of the 24 joints, the RMS change of its 300-frame
    trajectory between the best individual of consecutive generations."""
    import matplotlib.pyplot as plt

    gaits = summary["best_gait_per_gen"]  # (gens, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    change = np.sqrt(np.mean(np.diff(gaits, axis=0) ** 2, axis=1))  # (gens-1, NUM_LEGS, NUM_LEG_SEGS)
    change = change.reshape(change.shape[0], -1).T                  # (24, gens-1)
    change = np.column_stack([np.zeros(NUM_LEGS * NUM_LEG_SEGS), change])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(change, aspect="auto", cmap="viridis")
    ax.set(xlabel="Generation", ylabel="Joint",
           title="Joint evolution heatmap: RMS trajectory change of the generation best")
    ax.set_yticks(range(24), JOINT_LABELS, fontsize=7)
    fig.colorbar(im, ax=ax, label="RMS change (rad)")
    fig.tight_layout()
    path = out / "graph2_joint_evolution_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_joint_diversity(summary: dict, out: Path) -> Path:
    """Graph 3: population diversity per joint and generation - the RMS (over
    the 300 frames) of the population's standard deviation at each frame.
    Reveals joints whose variation has collapsed (converged) and joints
    still exploring.

    Our chromosome encodes angles directly (no baseline/amplitude/phase
    parameters to separate), so this is a single heatmap rather than the
    three-panel version used for a sinusoidal-parameter encoding."""
    import matplotlib.pyplot as plt

    population = summary["population_per_gen"]  # (gens, pop, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    per_frame_std = population.std(axis=1)                      # (gens, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    diversity = np.sqrt(np.mean(per_frame_std ** 2, axis=1))     # (gens, NUM_LEGS, NUM_LEG_SEGS)
    diversity = diversity.reshape(diversity.shape[0], -1).T      # (24, gens)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(diversity, aspect="auto", cmap="magma")
    ax.set(xlabel="Generation", ylabel="Joint",
           title="Joint diversity across the population over generations")
    ax.set_yticks(range(24), JOINT_LABELS, fontsize=7)
    fig.colorbar(im, ax=ax, label="Population std, RMS over frames (rad)")
    fig.tight_layout()
    path = out / "graph3_joint_diversity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_initial_vs_evolved(summary: dict, out: Path) -> Path:
    """Graph 4: 300-frame trajectories of all 24 joints - best of generation 0
    vs the best-ever evolved individual."""
    import matplotlib.pyplot as plt

    initial = summary["best_gait_per_gen"][0]
    evolved = summary["best_ever_gait"]

    fig, axes = plt.subplots(NUM_LEGS, NUM_LEG_SEGS, figsize=(14, 16), sharex=True)
    for leg in range(NUM_LEGS):
        for joint in range(NUM_LEG_SEGS):
            ax = axes[leg, joint]
            ax.plot(initial[:, leg, joint], color="grey", lw=1, label="generation 0 best")
            ax.plot(evolved[:, leg, joint], color="tab:blue", lw=1, label="best ever")
            if leg == 0:
                ax.set_title(JOINT_NAMES[joint])
            if joint == 0:
                ax.set_ylabel(LEG_LABELS[leg])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Initial vs evolved gait: joint-angle trajectories (rad) over 300 frames")
    fig.tight_layout()
    path = out / "graph4_initial_vs_evolved.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
