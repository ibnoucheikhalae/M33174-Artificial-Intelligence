import numpy as np
from numpy.typing import NDArray

from GA import (
    NUM_FRAMES,
    NUM_LEG_SEGS,
    NUM_LEGS,
    MIN_JOINT_ANGLES,
    MAX_JOINT_ANGLES,
)

from spider_kinematics import calculate_foot_positions

POSE_SHAPE = (NUM_LEGS, NUM_LEG_SEGS)
GAIT_SHAPE = (NUM_FRAMES, *POSE_SHAPE)

ANGLE_RANGES = MAX_JOINT_ANGLES - MIN_JOINT_ANGLES

FRONT_LEGS = [0, 1, 6, 7]  # L1, L2, R2, R1
BACK_LEGS = [2, 3, 4, 5]  # L3, L4, R3, R4
MIN_END_FEET = 2  # planted feet needed at EACH end

LEFT_LEGS = [0, 1, 2, 3]
RIGHT_MIRRORED = [7, 6, 5, 4]

MIRROR_SIGNS = np.asarray((-1.0, 1.0, 1.0), dtype=np.float64)

TIME_AXIS = -3
ALL_AXES = (-3, -2, -1)


START = np.array([[0.0, -1.2, -0.7]] * 8, dtype=np.float64)


GROUND_Z = -0.8194
GROUND_CONTACT_TOLERANCE = 0.05
MIN_CONTACT_FEET = 4
FORWARD_AXIS = 0  # walk along x axis
FORWARD_SIGN = 1.0  # wlak direction
FORWARD_DISTANCE_SCALE = 12


AIR_Z_LIMIT = 0.3  # max height a foot may lift above the ground

INVALID_FITNESS = -10.0  # used for invalid foot positions

BASE_FITNESS = 1.0

STEP_WEIGHT = 0.4
SMOOTHNESS_WEIGHT = 0.7
SYMMETRY_WEIGHT = 0.4
STRIDE_WEIGHT = 0.3
POSTURE_WEIGHT = 0.8
START_WEIGHT = 0.6
HEIGHT_WEIGHT = 1.0
LOOP_WEIGHT = 0.6

FORWARD_WEIGHT = 4.0
BACKWARD_WEIGHT = 0.5
CONTACT_WEIGHT = 1.0

MAX_STEP = 0.04  # Speed limit on each joint
MAX_ACCEL = 0.1  # checking the rate of change in max_step

SYMMETRY_TOL = 0.1
POSTURE_TOL = np.array([0.06, 0.2, 0.2], dtype=np.float64)
START_TOL = 0.05
BODY_SHIFT_TOL = 0.05  # how far the leg is fromn the ground

MIN_HIP_SWING = 0.20

HIP = 0  # index of th ehip joint


# used to return zero when its with its tolerance an dgrowing penalty when it goes over
def _excess(
    value: NDArray[np.float64],
    tolerance: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        (np.maximum(0.0, value - tolerance) / tolerance) ** 2, dtype=np.float64
    )


def _validate_angles(gait: NDArray[np.float64]) -> None:
    if gait.shape[-3:] != GAIT_SHAPE:
        raise ValueError(f"Expected shape (..., {GAIT_SHAPE}), got {gait.shape}")

    if not np.isfinite(gait).all():
        raise ValueError("Angles must be finite")

    if (gait < MIN_JOINT_ANGLES).any() or (gait > MAX_JOINT_ANGLES).any():
        raise ValueError("Angles must be within joint limits")


# makes a gait a circle instead of a line, used for the step_size_penalty and smoothess_penalty
def _wrap(gait: NDArray[np.float64], extra: int) -> NDArray[np.float64]:
    return np.concatenate([gait, gait[..., :extra, :, :]], axis=-3)


def step_size_penalty(gait: NDArray[np.float64]):
    step = np.abs(np.diff(_wrap(gait, 1), axis=TIME_AXIS)) / ANGLE_RANGES
    mean_part = np.mean(_excess(step, MAX_STEP), axis=ALL_AXES)
    worst = np.max(np.maximum(0.0, step - MAX_STEP) / MAX_STEP, axis=ALL_AXES)
    return mean_part + 0.1 * worst


def smoothness_penalty(gait: NDArray[np.float64]):
    accel = np.abs(np.diff(_wrap(gait, 2), n=2, axis=TIME_AXIS)) / ANGLE_RANGES
    return np.mean(_excess(accel, MAX_ACCEL), axis=ALL_AXES)


def symmetry_penalty(gait: NDArray[np.float64]):
    left = gait[..., LEFT_LEGS, :]
    mirrored_right = gait[..., RIGHT_MIRRORED, :] * MIRROR_SIGNS
    error = np.abs(left - mirrored_right) / ANGLE_RANGES[LEFT_LEGS]
    return np.mean(_excess(error, SYMMETRY_TOL), axis=ALL_AXES)


def stride_penalty(gait: NDArray[np.float64]):
    hip = gait[..., HIP]  # (..., T, 8)
    swing = (hip.max(axis=-2) - hip.min(axis=-2)) / ANGLE_RANGES[:, HIP]
    shortfall = np.maximum(0.0, MIN_HIP_SWING - swing) / MIN_HIP_SWING
    return np.mean(shortfall**2, axis=-1)


def posture_penalty(gait: NDArray[np.float64]):
    dev = np.abs(gait - START) / ANGLE_RANGES
    return np.mean(_excess(dev, POSTURE_TOL), axis=ALL_AXES)


EDGE_FRAMES = 5


def start_pose_penalty(gait: NDArray[np.float64]):
    error = np.abs(gait[..., :EDGE_FRAMES, :, :] - START) / ANGLE_RANGES
    return np.mean(_excess(error, START_TOL), axis=(-3, -2, -1))


def loop_penalty(gait: NDArray[np.float64]):
    error = np.abs(gait[..., -EDGE_FRAMES:, :, :] - START) / ANGLE_RANGES
    return np.mean(_excess(error, START_TOL), axis=(-3, -2, -1))


def _movement_batch(gaits: NDArray[np.float64]):
    """
    Movement metrics for a population (N, T, 8, 3).
    Returns displacement (N,), contact_ratio (N,), supported_intervals (N,),
    finite (N,), height_pen (N,).
    """
    foot_positions = calculate_foot_positions(gaits)  # (N, T, 8, 3)
    finite = np.isfinite(foot_positions).all(axis=(1, 2, 3))

    # penalise feet floating too high or sinking below the floor
    foot_z = foot_positions[..., 2]  # (N, T, 8)
    lowest = foot_z.min(axis=-1, keepdims=True)  # (N, T, 1) the foot the body rests on
    rel = foot_z - lowest  # height above that foot

    # Body riding up or sagging: the lowest foot should sit at the standing floor level
    body_shift = np.abs(lowest[..., 0] - GROUND_Z)  # (N, T)
    shift_pen = np.mean(_excess(body_shift, BODY_SHIFT_TOL), axis=1)

    # Feet held too high above the floor
    air = np.max(np.maximum(0.0, rel - AIR_Z_LIMIT) ** 2, axis=(1, 2))
    height_pen = air + shift_pen

    # A foot touches if it is near the lowest foot, not near a fixed number
    ground_contact = rel <= GROUND_CONTACT_TOLERANCE

    planted = ground_contact[:, :-1] & ground_contact[:, 1:]
    counts = planted.sum(axis=-1)

    # Gradual reward for having grounded feet
    contact_score = np.mean(
        np.minimum(counts, MIN_CONTACT_FEET) / MIN_CONTACT_FEET,
        axis=1,
    )

    supported = (
        (counts >= MIN_CONTACT_FEET)
        & (planted[..., FRONT_LEGS].sum(axis=-1) >= MIN_END_FEET)
        & (planted[..., BACK_LEGS].sum(axis=-1) >= MIN_END_FEET)
    )

    foot_disp = (
        foot_positions[:, 1:, :, FORWARD_AXIS] - foot_positions[:, :-1, :, FORWARD_AXIS]
    )

    vals = np.where(planted, -FORWARD_SIGN * foot_disp, np.inf)
    vals.sort(axis=-1)
    last = vals.shape[-1] - 1
    lo = np.clip((counts - 1) // 2, 0, last)[..., None]
    hi = np.clip(counts // 2, 0, last)[..., None]
    median = (
        0.5
        * (
            np.take_along_axis(vals, lo, axis=-1)
            + np.take_along_axis(vals, hi, axis=-1)
        )[..., 0]
    )

    frame_disp = np.where(supported, median, 0.0)

    total_intervals = gaits.shape[1] - 1
    displacement = frame_disp.sum(axis=1)
    supported_count = supported.sum(axis=1)
    contact_ratio = supported_count / total_intervals if total_intervals > 0 else 0.0

    return (
        displacement,
        contact_ratio,
        supported_count,
        finite,
        height_pen,
        contact_score,
    )


def forward_movement_reward(displacement: float):
    return np.tanh(np.maximum(0.0, displacement) / FORWARD_DISTANCE_SCALE)


def backward_movement_penalty(displacement: float):
    return np.tanh(np.maximum(0.0, -displacement) / FORWARD_DISTANCE_SCALE)


def fitness_function_batch(gaits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Evaluate a whole population at once.

    Input:  (N, 300, 8, 3), radians
    Output: (N,) fitness values, higher is better
    """
    _validate_angles(gaits)

    displacement, _, _, finite, height_pen, contact_score = _movement_batch(gaits)

    fitness = (
        BASE_FITNESS
        + FORWARD_WEIGHT * forward_movement_reward(displacement)
        - BACKWARD_WEIGHT * backward_movement_penalty(displacement)
        + CONTACT_WEIGHT * contact_score
        - STEP_WEIGHT * step_size_penalty(gaits)
        - SMOOTHNESS_WEIGHT * smoothness_penalty(gaits)
        - SYMMETRY_WEIGHT * symmetry_penalty(gaits)
        - STRIDE_WEIGHT * stride_penalty(gaits)
        - POSTURE_WEIGHT * posture_penalty(gaits)
        - START_WEIGHT * start_pose_penalty(gaits)
        - HEIGHT_WEIGHT * height_pen  # NEW
        - LOOP_WEIGHT * loop_penalty(gaits)
    )

    return np.where(finite, fitness, INVALID_FITNESS)
