import numpy as np
from numpy.typing import NDArray
from GA import (
    NUM_FRAMES,
    NUM_LEG_SEGS,
    NUM_LEGS,
    MIN_JOINT_ANGLES,
    MAX_JOINT_ANGLES,
)

POSE_SHAPE = (NUM_LEGS, NUM_LEG_SEGS)
GAIT_SHAPE = (NUM_FRAMES, *POSE_SHAPE)

ANGLE_RANGES = MAX_JOINT_ANGLES - MIN_JOINT_ANGLES

# Mirror-pair indexing: left leg i pairs with right leg RIGHT_MIRRORED[i].
LEFT_LEGS = [0, 1, 2, 3]  # L1..L4
RIGHT_MIRRORED = [7, 6, 5, 4]  # R1..R4, so index i pairs with left i

# How a right-leg angle maps into the left-leg convention:
#   expected_left = right * MIRROR_SIGNS + MIRROR_OFFSETS
# All three joints flip sign (confirmed by the real midpoint pose: L1 vs R1 are
# exact negatives), and the left/right joint limits are negations of each other.
MIRROR_SIGNS = np.asarray((-1.0, -1.0, -1.0), dtype=np.float64)
MIRROR_OFFSETS = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)

# Target poses extracted from the Webots controller.
# Leg order: L1, L2, L3, L4, R4, R3, R2, R1
# Joint order: hip, thigh, knee
# All angles are in radians.

# Start / end pose
SPREAD = np.array(
    [
        [-0.48, -1.3056, 0.8156],  # L1
        [-0.20, -1.3056, 0.8156],  # L2
        [0.20, -1.3056, 0.8156],  # L3
        [0.48, -1.3056, 0.8156],  # L4
        [-0.48, 1.3056, -0.8156],  # R4
        [-0.20, 1.3056, -0.8156],  # R3
        [0.20, 1.3056, -0.8156],  # R2
        [0.48, 1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

# Mid pose
CLOSE = np.array(
    [
        [0.24, -1.3056, 0.8156],  # L1
        [0.10, -1.3056, 0.8156],  # L2
        [-0.10, -1.3056, 0.8156],  # L3
        [-0.24, -1.3056, 0.8156],  # L4
        [0.24, 1.3056, -0.8156],  # R4
        [0.10, 1.3056, -0.8156],  # R3
        [-0.10, 1.3056, -0.8156],  # R2
        [-0.24, 1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

# One closed cycle: SPREAD -> CLOSE -> SPREAD. With 300 poses the last index is 299,
# so the outward phase spans 150 frame intervals and the return spans 149.
TARGET_POSES = np.stack([SPREAD, CLOSE, SPREAD])
KEYFRAMES = np.array([0, 150, 299])

TARGET_STEP_SIZES = np.abs(CLOSE - SPREAD) / 150.

#* AI-Provided Formula for cubic interpolation *#
# Linear phase u runs 0 -> 1 -> 0 through the keyframes. Each half uses its own
# duration, so both the midpoint and the final pose are reached exactly.
_reference_phase = np.interp(np.arange(NUM_FRAMES), KEYFRAMES, [0.0, 1.0, 0.0])
# Smoothstep s(u) = 3u^2 - 2u^3 has s(0)=0, s(1)=1 and s'(u)=6u(1-u).
# Its zero endpoint derivatives make the continuous reference stop at the start,
# turnaround and end. It remains monotonic within each half, without overshoot.
_reference_progress = 3 * _reference_phase**2 - 2 * _reference_phase**3
# pose = start + progress * displacement. The extra axes broadcast one progress
# value per frame over all 8 legs and 3 joints, producing shape (300, 8, 3).
# This is a scoring reference; the GA still evolves every frame independently.
INTERPOLATED_REFERNCE_GAIT = (
    SPREAD + _reference_progress[:, None, None] * (CLOSE - SPREAD)
)

# Fitness = BASE_FITNESS - (weighted sum of penalties). A perfect gait scores
# BASE_FITNESS; every penalty subtracts from it. Weights: tune later.
BASE_FITNESS = 1.0
TARGET_WEIGHT = 0.5
STEP_SIZE_WEIGHT = 0.3
SYMMETRY_WEIGHT = 0.1
ACCELERATION_WEIGHT = 0.1


# --- Validation ------------------------------------------------------------
def _validate_angles(angles: NDArray[np.float64], shape: tuple[int, ...]) -> None:
    if angles.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {angles.shape}")
    if not np.isfinite(angles).all():
        raise ValueError("Angles must be finite")
    if (angles < MIN_JOINT_ANGLES).any() or (angles > MAX_JOINT_ANGLES).any():
        raise ValueError("Angles must be within joint limits")


# --- Reward / penalty helpers (each unit-testable on its own) ---------------
def target_gait_penalty(
    gait: NDArray[np.float64], target: NDArray[np.float64] = INTERPOLATED_REFERNCE_GAIT
) -> float:
    """Root-mean-square of range-normalised joint errors across the whole gait.

    Zero means every frame matches the target. Taking the square root keeps
    the penalty on the scale of the normalised angular errors.
    """
    _validate_angles(gait, GAIT_SHAPE)
    _validate_angles(target, GAIT_SHAPE)
    error = (gait - target) / ANGLE_RANGES
    return float(np.sqrt(np.mean(error**2)))


def acceleration_penalty(gait: NDArray[np.float64]) -> float:
    """
    Smoothness measure. Ensures step size is consistent between frames.

    Mean squared second difference penalises abrupt changes in velocity.
    """
    _validate_angles(gait, GAIT_SHAPE)
    second_diff = np.diff(gait, n=2, axis=0)
    return float(np.mean((second_diff / (2 * ANGLE_RANGES)) ** 2))


def step_size_penalty(
    gait: NDArray[np.float64],
    target: NDArray[np.float64] = INTERPOLATED_REFERNCE_GAIT,
) -> float:
    """Mean squared, range-normalised error from signed target steps.

    First differences give the angular change per frame. Keeping their sign
    penalises movement in the wrong direction as well as at the wrong speed.
    """
    _validate_angles(gait, GAIT_SHAPE)
    _validate_angles(target, GAIT_SHAPE)
    target_step_sizes = np.diff(target, axis=0)
    step_sizes = np.diff(gait, axis=0)
    error = (step_sizes - target_step_sizes) / ANGLE_RANGES
    return float(np.mean(error**2))


def symmetry_penalty(
    gait: NDArray[np.float64],
    mirror_signs: NDArray[np.float64] = MIRROR_SIGNS,
    mirror_offsets: NDArray[np.float64] = MIRROR_OFFSETS,
) -> float:
    """Left/right mirror symmetry; 0 means perfectly mirrored.

    gait shape (300, 8, 3), radians. Stored order: L1, L2, L3, L4, R4, R3, R2, R1.
    Each right leg is mapped into the left-leg convention
    (right * mirror_signs + mirror_offsets) and compared with its left partner
    (L1<->R1, L2<->R2, L3<->R3, L4<->R4) frame by frame. Errors are scaled by
    each joint's permitted range, then squared and averaged.
    """
    _validate_angles(gait, GAIT_SHAPE)

    if mirror_signs.shape != (NUM_LEG_SEGS,) or mirror_offsets.shape != (NUM_LEG_SEGS,):
        raise ValueError(
            f"mirror_signs and mirror_offsets must each have {NUM_LEG_SEGS} values"
        )
    if not np.isfinite(mirror_signs).all():
        raise ValueError("Signs must be finite")
    if not np.isfinite(mirror_offsets).all():
        raise ValueError("Offset must be finite")

    left = gait[:, LEFT_LEGS, :]
    right = gait[:, RIGHT_MIRRORED, :]

    expected_left = right * mirror_signs + mirror_offsets
    error = (left - expected_left) / ANGLE_RANGES[LEFT_LEGS]
    return float(np.mean(error**2))


# --- Combined fitness ---------------------------------------------------------
def fitness_function(
    gait: NDArray[np.float64],
    targets: NDArray[np.float64] = INTERPOLATED_REFERNCE_GAIT,
) -> float:
    """Fitness of one individual (a single scalar, higher is better).

    gait: (300, 8, 3) joint angles in radians.
    targets: (300, 8, 3) reference gait compared against every frame.
    fitness = BASE_FITNESS
            - TARGET_WEIGHT     * target_gait_penalty
            - STEP_SIZE_WEIGHT  * step_size_penalty
            - SYMMETRY_WEIGHT   * symmetry_penalty
            - ACCELERATION_WEIGHT * acceleration_penalty

    The best possible score is BASE_FITNESS; the result can be negative.
    """
    return float(
        BASE_FITNESS
        - TARGET_WEIGHT * target_gait_penalty(gait, targets)
        - SYMMETRY_WEIGHT * symmetry_penalty(gait)
        - STEP_SIZE_WEIGHT * step_size_penalty(gait, targets)
        - ACCELERATION_WEIGHT * acceleration_penalty(gait)
    )
