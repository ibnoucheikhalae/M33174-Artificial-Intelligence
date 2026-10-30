import numpy as np
from numpy.typing import NDArray
 
# --- Constants -------------------------------------------------------------
NUM_LEGS = 8
NUM_LEG_SEGS = 3
NUM_FRAMES = 300
 
# Joint limits (radians), copied from the GA file so both modules agree.
# Knee is flipped between sides because the Webots results were flipped.
RIGHT_MIN = [-1.57, -1.57, -2.0]
RIGHT_MAX = [1.57, 1.57, 1.0]
LEFT_MIN = [-1.57, -1.57, -1.0]
LEFT_MAX = [1.57, 1.57, 2.0]
 
# Leg order: L1, L2, L3, L4, R4, R3, R2, R1 (matches the controller).
MIN_JOINT_ANGLES = np.array([LEFT_MIN] * 4 + [RIGHT_MIN] * 4)
MAX_JOINT_ANGLES = np.array([LEFT_MAX] * 4 + [RIGHT_MAX] * 4)
 
POSE_SHAPE = (NUM_LEGS, NUM_LEG_SEGS)
GAIT_SHAPE = (NUM_FRAMES, *POSE_SHAPE)
KEYFRAMES = np.array([0, NUM_FRAMES // 2, NUM_FRAMES - 1])  # frames 0, 150, 299
ANGLE_RANGES = MAX_JOINT_ANGLES - MIN_JOINT_ANGLES
 
# Mirror-pair indexing: left leg i pairs with right leg RIGHT_MIRRORED[i].
LEFT_LEGS = [0, 1, 2, 3]          # L1..L4
RIGHT_MIRRORED = [7, 6, 5, 4]     # R1..R4, so index i pairs with left i
 
# How a right-leg angle maps into the left-leg convention:
#   expected_left = right * MIRROR_SIGNS + MIRROR_OFFSETS
# All three joints flip sign (confirmed by the real midpoint pose: L1 vs R1 are
# exact negatives), and the left/right joint limits are negations of each other.
MIRROR_SIGNS = (-1.0, -1.0, -1.0)
MIRROR_OFFSETS = (0.0, 0.0, 0.0)
 
# Target poses. START/END are still stubs until Prince/Alae finalise them;
# MIDPOINT is the real pose. Leg order: L1, L2, L3, L4, R4, R3, R2, R1.
START = np.zeros(POSE_SHAPE)                    # TODO: real start pose
MIDPOINT = np.array([
    [ 0.24, -1.3056,  0.8156],  # L1
    [ 0.10, -1.3056,  0.8156],  # L2
    [-0.10, -1.3056,  0.8156],  # L3
    [-0.24, -1.3056,  0.8156],  # L4
    [ 0.24,  1.3056, -0.8156],  # R4
    [ 0.10,  1.3056, -0.8156],  # R3
    [-0.10,  1.3056, -0.8156],  # R2
    [-0.24,  1.3056, -0.8156],  # R1
], dtype=float)
MIDPOINT_VECTOR = MIDPOINT.flatten()            # 24 values, row-major (leg by leg)
END = START.copy()                              # TODO: real end pose
STUB_TARGETS = np.stack([START, MIDPOINT, END])  # (3, 8, 3)
 
# Fitness = BASE_FITNESS - (weighted sum of penalties). A perfect gait scores
# BASE_FITNESS; every penalty subtracts from it. Weights: tune later.
BASE_FITNESS = 1.0
TARGET_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.2
SYMMETRY_WEIGHT = 0.2
JERK_WEIGHT = 0.0  # wired in but switched off
 
 
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
    _validate_angles(targets, (3, *POSE_SHAPE))
    error = (gait[KEYFRAMES] - targets) / ANGLE_RANGES
    return float(np.sqrt(np.mean(error**2)))
 
 
def smoothness_penalty(gait: NDArray[np.float64]) -> float:
    """Std of per-frame change over time for each joint separately, then
    averaged over all 24 joints. A joint moving at constant speed scores 0,
    and different joints moving at different speeds don't affect each other.
    """
    _validate_angles(gait, GAIT_SHAPE)
    changes = np.diff(gait, axis=0)               # (299, 8, 3)
    per_joint = np.std(changes, axis=0, ddof=0)   # (8, 3)
    return float(np.mean(per_joint))
 
 
def jerk_penalty(gait: NDArray[np.float64]) -> float:
    """Alternative smoothness measure (weight is 0 by default).
 
    Mean squared second difference penalises abrupt changes in velocity.
    """
    _validate_angles(gait, GAIT_SHAPE)
    return float(np.mean(np.diff(gait, n=2, axis=0) ** 2))
 
 
def symmetry_penalty(
    gait: NDArray[np.float64],
    mirror_signs: tuple[float, float, float] = MIRROR_SIGNS,
    mirror_offsets: tuple[float, float, float] = MIRROR_OFFSETS,
) -> float:
    """Left/right mirror symmetry; 0 means perfectly mirrored.
 
    gait shape (300, 8, 3), radians. Stored order: L1, L2, L3, L4, R4, R3, R2, R1.
    Each right leg is mapped into the left-leg convention
    (right * mirror_signs + mirror_offsets) and compared with its left partner
    (L1<->R1, L2<->R2, L3<->R3, L4<->R4) frame by frame. Errors are scaled by
    each joint's permitted range, then squared and averaged.
    """
    _validate_angles(gait, GAIT_SHAPE)
    signs = np.asarray(mirror_signs, dtype=np.float64)
    offsets = np.asarray(mirror_offsets, dtype=np.float64)
    if signs.shape != (NUM_LEG_SEGS,) or offsets.shape != (NUM_LEG_SEGS,):
        raise ValueError(
            f"mirror_signs and mirror_offsets must each have {NUM_LEG_SEGS} values"
        )
 
    left = gait[:, LEFT_LEGS, :]
    right = gait[:, RIGHT_MIRRORED, :]
 
    expected_left = right * signs + offsets
    error = (left - expected_left) / ANGLE_RANGES[LEFT_LEGS]
    return float(np.mean(error**2))
 
 
# --- Combined fitness ---------------------------------------------------------
def fitness_function(
    gait: NDArray[np.float64],
    targets: NDArray[np.float64] = STUB_TARGETS,
) -> float:
    """Fitness of one individual (a single scalar, higher is better).
 
    gait: (300, 8, 3) joint angles in radians.
    targets: (3, 8, 3) target poses compared against frames 0, 150, 299.
 
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
        - SMOOTHNESS_WEIGHT * smoothness_penalty(gait)
        - SYMMETRY_WEIGHT * symmetry_penalty(gait)
        - JERK_WEIGHT * jerk_penalty(gait)
    )
 