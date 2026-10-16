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
from numpy.typing import ArrayLike, NDArray


def plot_spider_pose(angles: ArrayLike) -> None:
    """Plot a static 3D spider pose based on joint angles.

    Input:
        angles: 1x24 vector of joint angles in radians
                [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
    Legs are arranged in this configuration: {'L1', 'L2', 'L3', 'L4','R4', 'R3', 'R2', 'R1'}
    """
    # Parameters
    n_legs = 8
    segment_lengths = np.array([1.2, 0.7, 1.0])  # [Coxa, Femur, Tibia]
    a = 1.5
    b = 1.0  # Ellipse axes for body (oval shape)

    # Base angles (L1 front-left to L4 rear-left, R4 rear-right to R1 front-right)
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))

    # Leg labels
    leg_labels = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]

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
            base_pos, angle, [theta1, theta2, theta3], segment_lengths
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


def forward_leg_kinematics2(
    base_pos: ArrayLike,
    base_angle: float,
    joint_angles: ArrayLike,
    segment_lengths: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the four joint positions of a three-segment leg.

    base_pos: [x,y,z] position of leg base on body
    base_angle: angle around body ellipse where leg base is located (radians)
    joint_angles: [theta1, theta2, theta3] joint angles for the leg in radians
    segment_lengths: [coxa, femur, tibia] lengths of leg segments
    """
    base_pos = np.asarray(base_pos, dtype=float).reshape(3)
    joint_angles = np.asarray(joint_angles, dtype=float).reshape(3)
    segment_lengths = np.asarray(segment_lengths, dtype=float).reshape(3)

    # Unpack joint angles
    theta1 = joint_angles[0]  # Coxa yaw (rotation about vertical axis)
    theta2 = joint_angles[1]  # Femur pitch
    theta3 = joint_angles[2]  # Tibia pitch

    # Unpack segment lengths
    L1 = segment_lengths[0]  # Coxa length
    L2 = segment_lengths[1]  # Femur length
    L3 = segment_lengths[2]  # Tibia length

    # Joint 1: leg base on body
    j1 = base_pos  # starting point

    # --- Compute Coxa direction with elevation ---
    coxa_elevation = np.deg2rad(30)  # fixed 30 degree upward pitch for coxa

    # Horizontal direction of coxa in XY plane based on base_angle + theta1
    coxa_horiz_dir = np.array([np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0])

    # Rotation axis for pitch up: perpendicular to coxa horizontal direction in XY plane
    rot_axis = np.cross(coxa_horiz_dir, [0, 0, 1])

    # Rotation matrix around rot_axis by coxa_elevation
    R = axis_angle_rotation_matrix(rot_axis, coxa_elevation)

    # Rotate horizontal coxa direction upward
    # With NumPy's 1D vectors, @ performs matrix-vector multiplication directly;
    # no row-to-column transpose or transpose back is needed.
    coxa_dir = R @ coxa_horiz_dir

    # Joint 2 position: end of coxa segment
    j2 = j1 + L1 * coxa_dir

    # --- Femur rotation ---
    # Femur pitch is relative to coxa direction, rotate in plane defined by coxa_dir
    # To simplify, rotate femur around axis perpendicular to coxa_dir and Z

    # Define femur rotation axis (perpendicular to coxa_dir and vertical axis)
    femur_rot_axis = np.cross(coxa_dir, [0, 0, 1])
    femur_rot_axis = femur_rot_axis / np.linalg.norm(femur_rot_axis)

    # Femur direction vector starts aligned with coxa_dir
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2)

    # Joint 3 position: end of femur segment
    j3 = j2 + L2 * femur_dir

    # --- Tibia rotation ---
    # Tibia pitch is relative to femur direction
    # Rotate tibia around axis perpendicular to femur_dir and vertical axis
    tibia_rot_axis = np.cross(femur_dir, [0, 0, 1])
    # Normalising the vector converts it to 1 whilst preserving it's direction
    # E.g., [0, -0.5, 0] / 0.5 ==> [0, -1, 0]
    #! A problem arises when the femur points exactly up or down. It is then parallel to the
    #! veritcal vector and so the cross product is 0.
    #! A zero vector has no direction. Dividing by its length means dividing by 0, producing NaN.
    #! Very close to vertical, the number is tiny, so floating point errors can affect direction.
    tibia_rot_axis = tibia_rot_axis / np.linalg.norm(tibia_rot_axis)

    # Tibia direction vector
    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3)

    # Joint 4 position: end of tibia segment (foot)
    j4 = j3 + L3 * tibia_dir
    return j1, j2, j3, j4


# --- Helper function: axis-angle rotation matrix ---
def axis_angle_rotation_matrix(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Build the original Rodrigues rotation matrix for an axis and angle in radians."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    # Normalization removes any dependence on axis length; a zero axis remains
    # undefined, just as in the MATLAB implementation.
    axis = axis / np.linalg.norm(axis)
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )
    return R


# --- Helper function: rotate a vector around an axis by an angle ---
def rotate_vector(v: ArrayLike, axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Rotate a vector around an axis by an angle in radians."""
    R = axis_angle_rotation_matrix(axis, angle)
    v_rot = R @ np.asarray(v, dtype=float).reshape(3)  # @ Performs matrix multiplication
    return v_rot


if __name__ == "__main__":
    # Example: replace these 24 values with your joint angles, in radians.
    angles = np.zeros(24)
    plot_spider_pose(angles)
    plt.show()
