"""Animate the best saved gait: python render_gait.py [path/to/results.pkl]."""

import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from plot_spider_pose import plot_spider_pose


def main() -> None:
    """Load the highest-scoring gait and play its poses at 30 frames per second."""
    path = Path(sys.argv[1])
    with path.open("rb") as file:
        results = pickle.load(file)
    config = max(results, key=lambda key: results[key]["best_fitness"].max())
    gait = results[config]["best_gait"]
    print("Best configuration:", config)
    print("Best fitness:", results[config]["best_fitness"].max())

    def draw_frame(frame: int) -> None:
        # The pose renderer clears figure 1 and prints angles on every call.
        with redirect_stdout(None):
            plot_spider_pose(gait[frame].reshape(24))
        plt.figure(1).suptitle(f"Frame {frame + 1}/{len(gait)}")

    figure = plt.figure(1)
    animation = FuncAnimation(figure, draw_frame, frames=len(gait), interval=1000 / 30)
    plt.show()
    # Keep the animation alive until the plot window closes.
    del animation


if __name__ == "__main__":
    sys.exit(main())