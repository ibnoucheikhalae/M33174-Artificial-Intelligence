import numpy as np
from numpy.typing import ArrayLike, NDArray


def forward_leg_kinematics2(
    base_pos: ArrayLike,
    base_angle: float,
    joint_angles: ArrayLike,
    segment_lengths: ArrayLike,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
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
    coxa_horiz_dir = np.array(
        [np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0]
    )

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
    axis_array = np.asarray(axis, dtype=np.float64).reshape(3)
    # Normalization removes any dependence on axis length; a zero axis remains
    # undefined, just as in the MATLAB implementation.
    axis_array = axis_array / np.linalg.norm(axis_array)
    x = axis_array[0]
    y = axis_array[1]
    z = axis_array[2]
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
    v_rot = R @ np.asarray(v, dtype=float).reshape(
        3
    )  # @ Performs matrix multiplication
    return v_rot


def calculate_foot_positions_slow(
    gait: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the 3D foot position of every leg across every gait frame.

    Input:
        gait: (frames, 8, 3)

    Returns:
        foot_positions: (frames, 8, 3)
    """

    segment_lengths = np.array([1.2, 0.7, 1.0])

    a = 1.5
    b = 1.0

    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])

    base_angles = np.concatenate((left_leg_angles, right_leg_angles))

    foot_positions = np.empty((gait.shape[0], 8, 3))

    for frame_index, frame in enumerate(gait):
        for leg_index in range(8):
            base_angle = base_angles[leg_index]

            base_pos = np.array([a * np.cos(base_angle), b * np.sin(base_angle), 0.0])

            _, _, _, foot = forward_leg_kinematics2(
                base_pos, base_angle, frame[leg_index], segment_lengths
            )

            foot_positions[frame_index, leg_index] = foot

    return foot_positions


SEGMENT_LENGTHS = np.array([1.2, 0.7, 1.0])
BODY_A = 1.5
BODY_B = 1.0
BASE_ANGLES = np.concatenate(
    (np.deg2rad([45, 75, 105, 135]), np.deg2rad([-135, -105, -75, -45]))
)
BASE_POS = np.stack(
    [BODY_A * np.cos(BASE_ANGLES), BODY_B * np.sin(BASE_ANGLES), np.zeros(8)], axis=1
)  # (8, 3)
COXA_ELEVATION = np.deg2rad(30)
Z_AXIS = np.array([0.0, 0.0, 1.0])


def _rotate_batch(v: NDArray, axis: NDArray, angle: NDArray) -> NDArray:
    """Rodrigues rotation of v about axis by angle, for arrays of shape (..., 3)."""
    k = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    c = np.cos(angle)[..., None]
    s = np.sin(angle)[..., None]
    return (
        v * c + np.cross(k, v) * s + k * np.sum(k * v, axis=-1, keepdims=True) * (1 - c)
    )


def calculate_foot_positions(gait: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Vectorised forward kinematics.

    Input:  (..., 8, 3) joint angles. Works for one gait (frames, 8, 3)
            or a whole population (pop, frames, 8, 3).
    Output: (..., 8, 3) foot x, y, z positions.
    """
    theta1, theta2, theta3 = gait[..., 0], gait[..., 1], gait[..., 2]

    with np.errstate(invalid="ignore", divide="ignore"):
        # Coxa: horizontal direction tilted up by the fixed 30 degrees
        phi = BASE_ANGLES + theta1
        coxa_dir = np.stack(
            [
                np.cos(COXA_ELEVATION) * np.cos(phi),
                np.cos(COXA_ELEVATION) * np.sin(phi),
                np.full_like(phi, np.sin(COXA_ELEVATION)),
            ],
            axis=-1,
        )

        # Femur
        femur_axis = np.cross(coxa_dir, Z_AXIS)
        femur_dir = _rotate_batch(coxa_dir, femur_axis, theta2)

        # Tibia (NaN if the femur is exactly vertical, same as the original)
        tibia_axis = np.cross(femur_dir, Z_AXIS)
        tibia_dir = _rotate_batch(femur_dir, tibia_axis, theta3)

    return (
        BASE_POS
        + SEGMENT_LENGTHS[0] * coxa_dir
        + SEGMENT_LENGTHS[1] * femur_dir
        + SEGMENT_LENGTHS[2] * tibia_dir
    )
