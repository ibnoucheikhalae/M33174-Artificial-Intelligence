"""Animate a saved gait.

Usage:
    python render_gait.py recent_run.pkl                 # interactive window
    python render_gait.py recent_run.pkl --save out.gif  # save headlessly
    python render_gait.py recent_run.pkl --save out.gif --stride 3

Accepts either a pickle holding {"best_gait": (300, 8, 3) array, ...} or a
raw .npy gait array.
"""

import argparse
import pickle
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import numpy as np

def load_gait(path: Path) -> np.ndarray:
    """Load a (300, 8, 3) gait from a results pickle or a .npy file."""
    if path.suffix == ".npy":
        gait = np.load(path)
    else:
        with path.open("rb") as file:
            results = pickle.load(file)
        gait = np.asarray(results["best_gait"], dtype=float)
    # Tolerate an extra leading axis, e.g. a saved (1, 300, 8, 3) selection.
    if gait.ndim == 4 and gait.shape[0] == 1:
        gait = gait[0]
    if gait.ndim != 3 or gait.shape[1:] != (8, 3):
        raise ValueError(f"Expected a (frames, 8, 3) gait, got {gait.shape}")
    return gait


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results .pkl or gait .npy")
    parser.add_argument(
        "--save", type=Path, default=None, help="write a .gif/.mp4 instead of showing a window"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="render every k-th frame (speeds up saving)"
    )
    args = parser.parse_args()

    # A non-interactive backend must be chosen before pyplot is imported.
    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from GA.plot_spider_pose import plot_spider_pose

    gait = load_gait(args.results)
    frames = range(0, len(gait), args.stride)

    def draw_frame(frame: int) -> None:
        # The pose renderer clears figure 1 and prints angles on every call.
        with redirect_stdout(None):
            plot_spider_pose(gait[frame].reshape(24))
        plt.figure(1).suptitle(f"Frame {frame + 1}/{len(gait)}")

    figure = plt.figure(1)
    animation = FuncAnimation(figure, draw_frame, frames=frames, interval=1000 / 30)
    if args.save is not None:
        fps = max(1, round(30 / args.stride))
        animation.save(args.save, fps=fps)
        print(f"Saved {args.save}")
    else:
        plt.show()
    # Keep the animation alive until the plot window closes.
    del animation


if __name__ == "__main__":
    main()
