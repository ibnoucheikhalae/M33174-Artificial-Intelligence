"""Plot a static 3D spider pose using NumPy and Matplotlib.

Near 1:1 translation of the supplied MATLAB functions.
All joint angles are in radians; leg order is L1, L2, L3, L4, R4, R3, R2, R1.


Usage:
    from plot_spider_pose import plot_spider_pose
    import matplotlib.pyplot as plt

    plot_spider_pose([0.0] * 24)
    plt.show()

Running this file directly displays an example with all joint angles set to zero.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from GA.kinematics import (
    SOCKET_HEADINGS,
    BODY_A,
    BODY_B,
    LEG_LABELS,
    NUM_LEGS,
    SEGMENT_LENGTHS,
    forward_leg_kinematics2,
)


def plot_spider_pose(angles: ArrayLike) -> None:
    """Plot a static 3D spider pose based on joint angles.

    Input:
        angles: 1x24 vector of joint angles in radians
                [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
    Legs are arranged in this configuration: {'L1', 'L2', 'L3', 'L4','R4', 'R3', 'R2', 'R1'}
    """
    n_legs = NUM_LEGS
    segment_lengths = SEGMENT_LENGTHS
    a = BODY_A
    b = BODY_B
    base_angles = SOCKET_HEADINGS
    leg_labels = LEG_LABELS

    # Validate input
    # NumPy arrays allow elementwise arithmetic; accept a list, a flat vector,
    # or a MATLAB-style row/column vector, then use zero-based flat indexing.
    angles = np.asarray(angles, dtype=float)
    # Extra check to ensure we are using a 1D array, or a 2D array with one row or colum
    if angles.size != n_legs * 3 or not (
        angles.ndim == 1 or (angles.ndim == 2 and 1 in angles.shape)
    ):
        raise ValueError("Input angles must be a 1x24 vector (3 angles per leg for 8 legs).")
    angles = angles.reshape(-1)  # Squashes back into 1D array

    # Setup figure
    fig = plt.figure(1)
    fig.clear()
    fig.set_facecolor("w")
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True)
    # Matplotlib keeps successive plots on the same axes automatically (hold on).
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=45, azim=45)  # window view
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 2)
    # Match MATLAB's axis equal: the plot box follows the data spans (8, 8, 4),
    # so one unit has the same scale on each axis, including the shorter Z axis.
    ax.set_box_aspect((8, 8, 4))

    # Plot body (oval shape)
    t = np.linspace(0, 2 * np.pi, 100)
    body_x = a * np.cos(t)
    body_y = b * np.sin(t)
    ax.plot(body_x, body_y, np.zeros_like(t), "k-", linewidth=3)

    # Head marker (front of spider at +X)
    # Use single-element sequences: Matplotlib's 3D plot expects iterable coordinates.
    ax.plot([a + 0.2], [0], [0], "r^", markersize=10, markerfacecolor="r")

    # Print joint angles for all legs
    print("--- Spider Pose ---")

    # Loop over legs
    for i in range(n_legs):
        # Indices for this leg's angles
        # MATLAB starts at 1; Python starts at 0, hence idx = i * 3.
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

        print(
            f"Leg {leg_labels[i]}: theta1 = {theta1:.3f} rad, "
            f"theta2 = {theta2:.3f} rad, theta3 = {theta3:.3f} rad"
        )

        # Compute leg base position on body ellipse
        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])

        # Compute FK for this leg
        j1, j2, j3, j4 = forward_leg_kinematics2(
            base_pos, float(angle), [theta1, theta2, theta3], segment_lengths
        )

        # Plot leg segments
        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], "k-", linewidth=2)
        ax.plot([j2[0], j3[0]], [j2[1], j3[1]], [j2[2], j3[2]], "b-", linewidth=2)
        ax.plot([j3[0], j4[0]], [j3[1], j4[1]], [j3[2], j4[2]], "r-", linewidth=2)
        ax.plot([j4[0]], [j4[1]], [j4[2]], "ro", markersize=5, markerfacecolor="r")

        # Label leg
        offset = 0.2
        label_pos = base_pos + offset * np.array([np.cos(angle), np.sin(angle), 0])
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2] + 0.05,
            leg_labels[i],
            fontsize=12,
            fontweight="bold",
        )

    # Python has no hold off equivalent; plt.show() displays the completed figure.
    # Keep it in the caller so this function also works in notebooks and scripts
    # that save the figure or draw several poses before opening a window.


if __name__ == "__main__":
    # Example: replace these 24 values with your joint angles, in radians.
    angles = np.zeros(24)
    plot_spider_pose(angles)
    plt.show()
