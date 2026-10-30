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
KEYFRAMES = KEYFRAMES = np.array([0, 80, 200, 299])
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

# Target poses. START/END are still stubs until Prince/Alae finalise them;

# Target poses extracted from the Webots controller.
# Leg order: L1, L2, L3, L4, R4, R3, R2, R1
# Joint order: hip, thigh, knee
# All angles are in radians.

# Frame 0: Initial standing position
START = np.array([
    [0.00, -1.3056,  0.8156],  # L1
    [0.00, -1.3056,  0.8156],  # L2
    [0.00, -1.3056,  0.8156],  # L3
    [0.00, -1.3056,  0.8156],  # L4
    [0.00,  1.3056, -0.8156],  # R4
    [0.00,  1.3056, -0.8156],  # R3
    [0.00,  1.3056, -0.8156],  # R2
    [0.00,  1.3056, -0.8156],  # R1
], dtype=float)

# Frame 80: Legs fully spread
SPREAD = np.array([
    [-0.48, -1.3056,  0.8156],  # L1
    [-0.20, -1.3056,  0.8156],  # L2
    [ 0.20, -1.3056,  0.8156],  # L3
    [ 0.48, -1.3056,  0.8156],  # L4
    [-0.48,  1.3056, -0.8156],  # R4
    [-0.20,  1.3056, -0.8156],  # R3
    [ 0.20,  1.3056, -0.8156],  # R2
    [ 0.48,  1.3056, -0.8156],  # R1
], dtype=float)

# Frame 200: Legs brought closer into pairs
CLOSE = np.array([
    [ 0.24, -1.3056,  0.8156],  # L1
    [ 0.10, -1.3056,  0.8156],  # L2
    [-0.10, -1.3056,  0.8156],  # L3
    [-0.24, -1.3056,  0.8156],  # L4
    [ 0.24,  1.3056, -0.8156],  # R4
    [ 0.10,  1.3056, -0.8156],  # R3
    [-0.10,  1.3056, -0.8156],  # R2
    [-0.24,  1.3056, -0.8156],  # R1
], dtype=float)

# Frame 299: Hold the final close position
END = CLOSE.copy()

# Combine the four target poses
TARGET_POSES = np.stack([START, SPREAD, CLOSE, END])


# Fitness = BASE_FITNESS - (weighted sum of penalties). A perfect gait scores
# BASE_FITNESS; every penalty subtracts from it. Weights: tune later.
BASE_FITNESS = 1.0
TARGET_WEIGHT = 0.6
SMOOTHNESS_WEIGHT = 0.2
SYMMETRY_WEIGHT = 0.2


# --- Validation ------------------------------------------------------------
def _validate_angles(angles: NDArray[np.float64], shape: tuple[int, ...]) -> None:
    if angles.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {angles.shape}")
    if not np.isfinite(angles).all():
        raise ValueError("Angles must be finite")
    if (angles < MIN_JOINT_ANGLES).any() or (angles > MAX_JOINT_ANGLES).any():
        raise ValueError("Angles must be within joint limits")


# --- Reward / penalty helpers (each unit-testable on its own) ---------------
def target_pose_penalty(
    gait: NDArray[np.float64], targets: NDArray[np.float64]
) -> float:
    """How far the three keyframes (0, 150, 299) are from their targets.

    Root-mean-square of the range-normalised joint errors: 0 means every
    keyframe matches exactly. RMSE (rather than MSE) is used so the penalty
    is on a useful scale; with MSE a gait that completely ignores the
    midpoint pose would only score ~0.03.
    """
    _validate_angles(gait, GAIT_SHAPE)
    _validate_angles(targets, (len(KEYFRAMES), *POSE_SHAPE))
    error = (gait[KEYFRAMES] - targets) / ANGLE_RANGES
    return float(np.sqrt(np.mean(error**2)))


def smoothness_penalty(gait: NDArray[np.float64]) -> float:
    """Alternative smoothness measure (weight is 0 by default).

    Mean squared second difference penalises abrupt changes in velocity.
    """
    _validate_angles(gait, GAIT_SHAPE)
    second_diff = np.diff(gait, n=2, axis=0)
    return float(np.mean((second_diff / (2 * ANGLE_RANGES)) ** 2))


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
    targets: NDArray[np.float64] = TARGET_POSES,
) -> float:
    """Fitness of one individual (a single scalar, higher is better).

    gait: (300, 8, 3) joint angles in radians.
    targets: (4, 8, 3) target poses compared against frames 0, 80, 200, 299.
    fitness = BASE_FITNESS
            - TARGET_WEIGHT     * target_pose_penalty
            - SMOOTHNESS_WEIGHT * smoothness_penalty
            - SYMMETRY_WEIGHT   * symmetry_penalty
            - JERK_WEIGHT       * jerk_penalty

    The best possible score is BASE_FITNESS; the result can be negative.
    """
    return float(
        BASE_FITNESS
        - TARGET_WEIGHT * target_pose_penalty(gait, targets)
        - SYMMETRY_WEIGHT * symmetry_penalty(gait)
        - SMOOTHNESS_WEIGHT * smoothness_penalty(gait)
    )
