"""Animate the highest-fitness saved gait at 30 frames per second.

From the repository root::

    python -m GA.render_gait nn_results/generated_gaits.npy
    python -m GA.render_gait recent_run.pkl

NumPy files contain radians with shape (N, 300, 8, 3), or a single gait with
shape (300, 8, 3). GA summary pickles supply the same data under best_gait;
only load trusted pickle files. Fitness is recomputed using the current GA
fitness function. The printed zero-based index identifies the chosen gait
within the file. The phase NN exports one gait, at index zero.
"""

import argparse
import pickle
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from GA.GA import NUM_FRAMES, NUM_LEG_SEGS, NUM_LEGS, evaluate_fitness
from GA.plot_spider_pose import plot_spider_pose

FRAMES_PER_SECOND = 30
GAIT_SHAPE = (NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)


def load_best_gait(path: Path) -> tuple[NDArray[np.float64], int, float]:
    """Load and validate saved radians, returning the best gait, index, and fitness.

    Accept a .npy array or a GA summary .pkl containing best_gait. Single gaits
    are treated as one-element collections. Select the first on a fitness tie.
    Reject empty, malformed, or nonfinite data once at this loading boundary.
    """
    if path.suffix == ".npy":
        gaits = np.load(path, allow_pickle=False)
    elif path.suffix == ".pkl":
        with path.open("rb") as file:
            gaits = pickle.load(file)["best_gait"]
    else:
        raise ValueError("expected a .npy gait array or .pkl GA summary")
    gaits = np.asarray(gaits, dtype=np.float64)
    if gaits.shape == GAIT_SHAPE:
        gaits = gaits[None]  # Add a gait axis so single gaits and batches share one path.
    if gaits.ndim != 4 or gaits.shape[1:] != GAIT_SHAPE or len(gaits) == 0:
        raise ValueError("expected nonempty gaits with shape (N, 300, 8, 3)")
    if not np.isfinite(gaits).all():
        raise ValueError("gaits must contain finite angles")
    fitness = evaluate_fitness(gaits)
    index = int(np.argmax(fitness))  # Index of the largest score; higher fitness is better.
    return gaits[index], index, float(fitness[index])


def animate_gait(gait: NDArray[np.float64]) -> None:
    """Play a loaded gait of shape (300, 8, 3) until the plot window closes."""

    def draw_frame(frame: int) -> None:
        """Replace the displayed spider pose with the requested frame."""
        # The pose renderer clears figure 1 and prints angles on every call.
        with redirect_stdout(None):
            plot_spider_pose(gait[frame].reshape(-1))  # Flatten the pose into joint angles.
        plt.figure(1).suptitle(f"Frame {frame + 1}/{len(gait)}")

    figure = plt.figure(1)
    animation = FuncAnimation(
        figure, draw_frame, frames=len(gait), interval=1000 / FRAMES_PER_SECOND
    )
    plt.show()
    # Keep the animation alive until the plot window closes.
    del animation


def main(argv: Sequence[str] | None = None) -> None:
    """Choose the highest-fitness gait from a saved file and open its animation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help=".npy gaits or a trusted GA summary .pkl")
    args = parser.parse_args(argv)
    gait, index, fitness = load_best_gait(args.path)
    print(f"Playing gait at index {index}; GA fitness={fitness:.8f}", flush=True)
    animate_gait(gait)


if __name__ == "__main__":
    main()
