"""Shared spider geometry and forward kinematics.

Mostly generated with AI from the supplied MATLAB renderer, then checked
against that file so the 3D plot and the fitness function describe the same
robot. No matplotlib, no GA operators.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

NUM_LEGS = 8
LEG_LABELS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]

# Body oval and three-segment legs (same numbers as the supplied MATLAB plot).
BODY_A = 1.5
BODY_B = 1.0
SEGMENT_LENGTHS = np.array([1.2, 0.7, 1.0])  # coxa, femur, tibia
NUM_LEG_SEGS = int(SEGMENT_LENGTHS.shape[0])
COXA_ELEVATION = np.deg2rad(30)

# Socket headings around the body, storage order L1..L4, R4..R1.
SOCKET_HEADINGS = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
LEG_BASES = np.stack(
    [
        BODY_A * np.cos(SOCKET_HEADINGS),
        BODY_B * np.sin(SOCKET_HEADINGS),
        np.zeros(NUM_LEGS),
    ],
    axis=-1,
)

# Walk around the body so we can tell if two legs have crossed:
# R4, R3, R2, R1, L1, L2, L3, L4, then wrap back to R4.
HEADING_ORDER = [4, 5, 6, 7, 0, 1, 2, 3]
NOMINAL_GAPS = np.diff(
    SOCKET_HEADINGS[HEADING_ORDER], append=SOCKET_HEADINGS[4] + 2 * np.pi
)

_VERTICAL = np.array([0.0, 0.0, 1.0])


def rotate_about(
    vectors: ArrayLike, axes: ArrayLike, angles: ArrayLike
) -> NDArray[np.float64]:
    """Rotate vectors about axes by angles (radians). Works on one vector or a batch."""
    vectors = np.asarray(vectors, dtype=float)
    axes = np.asarray(axes, dtype=float)
    angles = np.asarray(angles, dtype=float)
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    k = axes / np.maximum(norms, 1e-12)
    c = np.cos(angles)[..., None]
    s = np.sin(angles)[..., None]
    rotated = (
        vectors * c
        + np.cross(k, vectors) * s
        + k * np.sum(k * vectors, axis=-1, keepdims=True) * (1.0 - c)
    )
    return np.where(norms > 1e-9, rotated, vectors)


def rotate_vector(v: ArrayLike, axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Rotate one 3D vector around an axis by an angle in radians."""
    return rotate_about(np.asarray(v, dtype=float).reshape(3), axis, angle)


def axis_angle_rotation_matrix(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Rodrigues rotation matrix for one axis and angle in radians."""
    identity = np.eye(3)
    return np.stack(
        [rotate_about(identity[:, i], axis, angle) for i in range(3)], axis=1
    )


def forward_leg_kinematics2(
    base_pos: ArrayLike,
    base_angle: float,
    joint_angles: ArrayLike,
    segment_lengths: ArrayLike = SEGMENT_LENGTHS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Four joint positions of one three-segment leg (base, coxa end, knee, foot)."""
    j1 = np.asarray(base_pos, dtype=float).reshape(3)
    joint_angles = np.asarray(joint_angles, dtype=float).reshape(3)
    lengths = np.asarray(segment_lengths, dtype=float).reshape(3)
    theta1, theta2, theta3 = joint_angles
    coxa_horiz = np.array(
        [np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0.0]
    )
    coxa_dir = rotate_about(coxa_horiz, np.cross(coxa_horiz, _VERTICAL), COXA_ELEVATION)
    j2 = j1 + lengths[0] * coxa_dir
    femur_axis = np.cross(coxa_dir, _VERTICAL)
    femur_dir = rotate_about(coxa_dir, femur_axis, theta2)
    j3 = j2 + lengths[1] * femur_dir
    tibia_axis = np.cross(femur_dir, _VERTICAL)
    if np.linalg.norm(tibia_axis) <= 1e-9:
        tibia_axis = femur_axis
    tibia_dir = rotate_about(femur_dir, tibia_axis, theta3)
    j4 = j3 + lengths[2] * tibia_dir
    return j1, j2, j3, j4


def foot_positions(gait: NDArray[np.float64]) -> NDArray[np.float64]:
    """Foot XYZ for every frame and leg. gait shape (..., 8, 3) -> (..., 8, 3)."""
    theta1 = gait[..., 0]
    theta2 = gait[..., 1]
    theta3 = gait[..., 2]
    yaw = SOCKET_HEADINGS + theta1
    coxa_horiz = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)
    coxa_dir = rotate_about(
        coxa_horiz,
        np.cross(coxa_horiz, _VERTICAL),
        np.full(yaw.shape, COXA_ELEVATION),
    )
    j2 = LEG_BASES + SEGMENT_LENGTHS[0] * coxa_dir
    femur_axis = np.cross(coxa_dir, _VERTICAL)
    femur_dir = rotate_about(coxa_dir, femur_axis, theta2)
    j3 = j2 + SEGMENT_LENGTHS[1] * femur_dir
    tibia_axis = np.cross(femur_dir, _VERTICAL)
    tibia_norm = np.linalg.norm(tibia_axis, axis=-1, keepdims=True)
    tibia_axis = np.where(tibia_norm > 1e-9, tibia_axis, femur_axis)
    tibia_dir = rotate_about(femur_dir, tibia_axis, theta3)
    return j3 + SEGMENT_LENGTHS[2] * tibia_dir


def foot_heights(gait: NDArray[np.float64]) -> NDArray[np.float64]:
    """Foot Z for every frame and leg."""
    return foot_positions(gait)[..., 2]
