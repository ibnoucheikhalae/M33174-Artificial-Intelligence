"""Penalty-based fitness for evolved spider gaits.

All constraint handling lives here. The genetic operators never clamp,
smooth or repair a candidate: a gait with out-of-range angles or crossed
legs is still EVALUATED, it just scores badly in proportion to how badly it
misbehaves. Selection then removes it. This gives evolution a gradient
("slightly out of range" beats "far out of range") that outright rejection
or clamping would destroy.

fitness = BASE_FITNESS
        + MOVEMENT_WEIGHT * movement_reward
        + STEP_WEIGHT * step_reward
        - sum over penalties of  weight * (raw_error / scale)

Every raw measure is normalised by the joint ranges (or the leg spacing) so
the scales are comparable; `scale` states how much raw error costs that
term's full weight. `fitness_breakdown` exposes each term for analysis.
"""

import numpy as np
from numpy.typing import NDArray
from GA.gait_constants import (
    NUM_FRAMES,
    MIN_JOINT_ANGLES,
    MAX_JOINT_ANGLES,
    KEYFRAMES,
    TARGET_POSES,
    COUPLES,
)
from GA.kinematics import (
    NUM_LEGS,
    NUM_LEG_SEGS,
    HEADING_ORDER,
    SOCKET_HEADINGS,
    NOMINAL_GAPS,
    foot_heights,
)

POSE_SHAPE = (NUM_LEGS, NUM_LEG_SEGS)
GAIT_SHAPE = (NUM_FRAMES, *POSE_SHAPE)

ANGLE_RANGES = MAX_JOINT_ANGLES - MIN_JOINT_ANGLES

# Mirror-pair indexing: left leg i pairs with right leg RIGHT_MIRRORED[i].
LEFT_LEGS = [0, 1, 2, 3]  # L1..L4
RIGHT_MIRRORED = [7, 6, 5, 4]  # R1..R4, so index i pairs with left i

# How a right-leg angle maps into the left-leg convention:
#   expected_left = right * MIRROR_SIGNS + MIRROR_OFFSETS
# The supplied renderer mirrors hip yaw; thigh and knee pitch use the same
# convention on both sides, as do the joint limits in GA.
MIRROR_SIGNS = np.asarray((-1.0, 1.0, 1.0), dtype=np.float64)
MIRROR_OFFSETS = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)

# Alternating tetrapod groups, matching the seed's phase offsets:
# A = L1, L3, R4, R2; B = L2, L4, R3, R1.
GROUP_A = [0, 2, 4, 6]
GROUP_B = [1, 3, 5, 7]

# Foot-height geometry of the standing SPREAD pose. A gait only "takes a
# step" if a foot rises clearly above this; 0.35 units is a visible lift in
# the renderer (about 40% of the tibia).
STANCE_Z = -0.893
TARGET_CLEARANCE = 0.35
AIRBORNE_CLEARANCE = 0.5 * TARGET_CLEARANCE
TARGET_HIP_SWING = 0.25  # radians of hip yaw while the foot is in the air

# Movement reward saturation: a joint whose angle varies with a standard
# deviation of this much (radians) over the gait earns full movement credit.
# A 0.2 rad amplitude sinusoid has std ~0.14, so this rewards clearly visible
# motion without pushing joints to extremes.
FULL_MOVEMENT_STD = 0.15

BASE_FITNESS = 1.0
MOVEMENT_WEIGHT = 0.3
STEP_WEIGHT = 1.0
# Theoretical upper bound: full rewards and zero penalties may not be jointly attainable.
MAX_FITNESS = BASE_FITNESS + MOVEMENT_WEIGHT + STEP_WEIGHT

# Each penalty term: (weight, scale). A raw error equal to `scale` costs
# `weight` fitness. These are soft scoring scales, not hard limits; fitness
# may go negative for badly misbehaving gaits.
#   joint_limit: RMS range-normalised violation. 2% of a joint's range
#                outside the limit across the gait costs 1.0 -- the strongest
#                term because it replaces clamping as the constraint.
#   leg_overlap: RMS gap-normalised crossing. Legs crossing by 5% of their
#                nominal spacing costs 0.8.
#   target_pose: RMS range-normalised keyframe error; 5% of range costs 1.0.
#   acceleration: RMS range-normalised second difference. Second differences
#                grow with frequency squared. The scale lets a smooth 4-cycle
#                step cost a modest ~0.2, while mutation spikes remain dear.
#   symmetry:    RMS range-normalised left/right mirror error; 10% costs 0.3.
#   couple:      RMS range-normalised thigh/knee error between ipsilateral
#                pairs (L1-L3, L2-L4, R1-R3, R2-R4). 8% of range costs 0.5.
#   support:     Mean fraction of a 3-foot shortage per frame. A pose with
#                fewer than 3 feet down would tip over; 3 down costs 0.

# dict -> definition: weight, scale
PENALTY_WEIGHTS_AND_SCALES = {
    "joint_limit": (1.0, 0.02),
    "leg_overlap": (0.8, 0.05),
    "target_pose": (1.0, 0.05),
    "acceleration": (0.25, 0.01),
    "symmetry": (0.3, 0.10),
    "couple": (0.5, 0.08),
    "support": (0.6, 0.25),
}


# --- Validation ------------------------------------------------------------
def _validate_angles(angles: NDArray[np.float64], shape: tuple[int, ...]) -> None:
    """Shape and finiteness only. Out-of-range angles are deliberately
    accepted: invalid candidates must be scoreable, not rejected here."""
    if angles.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {angles.shape}")
    if not np.isfinite(angles).all():
        raise ValueError("Angles must be finite")


# --- Reward / penalty helpers (each unit-testable on its own) ---------------
def joint_limit_penalty(gait: NDArray[np.float64]) -> float:
    """RMS range-normalised violation of the joint limits; 0 when in range.

    Measures HOW FAR outside the limits the gait goes, not just whether it
    does: an angle 1 degree out scores better than one 50 degrees out, so
    selection can walk candidates back inside the valid region.
    """
    _validate_angles(gait, GAIT_SHAPE)
    below = np.maximum(MIN_JOINT_ANGLES - gait, 0.0)
    above = np.maximum(gait - MAX_JOINT_ANGLES, 0.0)
    violation = (below + above) / ANGLE_RANGES
    return float(np.sqrt(np.mean(violation**2)))


def leg_overlap_penalty(gait: NDArray[np.float64]) -> float:
    """RMS spacing-normalised leg crossing; 0 when no legs cross.

    Each leg's heading is its socket heading plus hip yaw.
    Walking around the body (R4, R3, R2, R1, L1, L2, L3, L4, back to R4),
    those headings must stay in order; a negative gap between neighbours
    means two legs have crossed. Only actual crossings are penalised, in
    proportion to how far past each other the legs swing.
    """
    _validate_angles(gait, GAIT_SHAPE)
    headings = SOCKET_HEADINGS + gait[:, :, 0]
    ordered = headings[:, HEADING_ORDER]
    gaps = np.diff(ordered, axis=1)
    wrap_gap = ordered[:, :1] + 2 * np.pi - ordered[:, -1:]
    gaps = np.concatenate([gaps, wrap_gap], axis=1)
    crossing = np.maximum(-gaps, 0.0) / NOMINAL_GAPS
    return float(np.sqrt(np.mean(crossing**2)))


def target_pose_penalty(
    gait: NDArray[np.float64], targets: NDArray[np.float64] = TARGET_POSES
) -> float:
    """RMS range-normalised error at the task keyframes (0, 150, 299).

    Only the three required poses are scored; the 297 frames in between are
    left entirely to evolution. 0 means every keyframe matches exactly.
    """
    _validate_angles(gait, GAIT_SHAPE)
    _validate_angles(targets, (len(KEYFRAMES), *POSE_SHAPE))
    error = (gait[KEYFRAMES] - targets) / ANGLE_RANGES
    return float(np.sqrt(np.mean(error**2)))


def acceleration_penalty(gait: NDArray[np.float64]) -> float:
    """RMS range-normalised second difference: the jitter / jerkiness price.

    Unsmoothed mutation necessarily injects single-frame spikes; this term is
    how selection (rather than any manual blending) favours offspring whose
    mutations happen to cooperate with their neighbouring frames.
    """
    _validate_angles(gait, GAIT_SHAPE)
    second_diff = np.diff(gait, n=2, axis=0) / ANGLE_RANGES
    return float(np.sqrt(np.mean(second_diff**2)))


def symmetry_penalty(
    gait: NDArray[np.float64],
    mirror_signs: NDArray[np.float64] = MIRROR_SIGNS,
    mirror_offsets: NDArray[np.float64] = MIRROR_OFFSETS,
) -> float:
    """Left/right mirror symmetry; 0 means perfectly mirrored.

    Each right leg is mapped into the left-leg convention
    (right * mirror_signs + mirror_offsets) and compared with its left
    partner (L1<->R1, L2<->R2, L3<->R3, L4<->R4) frame by frame. Returns the
    RMS range-normalised error.
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
    return float(np.sqrt(np.mean(error**2)))


def couple_penalty(gait: NDArray[np.float64]) -> float:
    """Ipsilateral couples (L1-L3, L2-L4, R1-R3, R2-R4); 0 when they match.

    Only thigh and knee are compared. Hip rest angles differ by design
    (front vs rear legs yaw inward from opposite sides to hit CLOSE), so
    identical hip trajectories would fight the keyframes. Matching thigh
    and knee is what makes a couple look like one step, not two.
    """
    _validate_angles(gait, GAIT_SHAPE)
    first = gait[:, COUPLES[:, 0], 1:]
    second = gait[:, COUPLES[:, 1], 1:]
    error = (first - second) / ANGLE_RANGES[COUPLES[:, 0], 1:]
    return float(np.sqrt(np.mean(error**2)))


def support_penalty(gait: NDArray[np.float64]) -> float:
    """Mean shortage of supporting feet; 0 when every pose has at least 3 down.

    A foot counts as down if it is within AIRBORNE_CLEARANCE of the standing
    height. Fewer than 3 planted feet would tip the body over.
    """
    _validate_angles(gait, GAIT_SHAPE)
    down = foot_heights(gait) < STANCE_Z + AIRBORNE_CLEARANCE
    shortage = np.maximum(3 - down.sum(axis=1), 0) / 3.0
    return float(np.mean(shortage))


def step_reward(gait: NDArray[np.float64]) -> float:
    """How much the gait looks like taking steps, between 0 and 1.

    A hip-only spread/close scores ~0: the feet never leave the ground.
    Full credit needs three things at once:
      - every leg lifts by TARGET_CLEARANCE (a visible step height);
      - the two tetrapod groups step in antiphase, not all eight together;
      - each airborne foot also yaws at the hip (swing, not a vertical hop).
    """
    _validate_angles(gait, GAIT_SHAPE)
    heights = foot_heights(gait)
    clearance = np.maximum(heights - STANCE_Z, 0.0)
    peak = np.minimum(clearance.max(axis=0) / TARGET_CLEARANCE, 1.0)

    airborne = clearance > AIRBORNE_CLEARANCE
    group_a = airborne[:, GROUP_A].mean(axis=1)
    group_b = airborne[:, GROUP_B].mean(axis=1)
    # Cover: some group is in swing. Overlap: both groups swing at once.
    cover = float(np.mean(np.maximum(group_a, group_b)))
    overlap = float(np.mean(group_a * group_b))
    alternation = cover * (1.0 - overlap)

    hip = gait[:, :, 0]
    swing = np.empty(NUM_LEGS)
    for leg in range(NUM_LEGS):
        up = airborne[:, leg]
        if np.count_nonzero(up) < 2:
            swing[leg] = 0.0
        else:
            swing[leg] = min(float(np.ptp(hip[up, leg])) / TARGET_HIP_SWING, 1.0)

    return float(0.5 * peak.mean() + 0.3 * alternation + 0.2 * swing.mean())


def movement_reward(gait: NDArray[np.float64]) -> float:
    """Mean per-joint movement credit, between 0 (static) and 1 (all moving).

    Each of the 24 joints earns min(std over frames / FULL_MOVEMENT_STD, 1).
    This rewards EVERY joint for visibly moving -- a direct counterweight to
    the acceleration and target penalties, which a frozen gait satisfies for
    free, and the measurable check that no joint is unintentionally stuck.
    """
    _validate_angles(gait, GAIT_SHAPE)
    per_joint_std = gait.std(axis=0)
    return float(np.mean(np.minimum(per_joint_std / FULL_MOVEMENT_STD, 1.0)))


_PENALTY_FUNCTIONS = {
    "joint_limit": joint_limit_penalty,
    "leg_overlap": leg_overlap_penalty,
    "target_pose": target_pose_penalty,
    "acceleration": acceleration_penalty,
    "symmetry": symmetry_penalty,
    "couple": couple_penalty,
    "support": support_penalty,
}


# --- Combined fitness ---------------------------------------------------------
def fitness_breakdown(gait: NDArray[np.float64]) -> dict[str, float]:
    """Each term's WEIGHTED contribution to fitness (reward positive,
    penalties negative), for analysis and debugging."""
    breakdown = {
        "movement": MOVEMENT_WEIGHT * movement_reward(gait),
        "step": STEP_WEIGHT * step_reward(gait),
    }
    for name, (weight, scale) in PENALTY_WEIGHTS_AND_SCALES.items():
        breakdown[name] = -(weight / scale) * _PENALTY_FUNCTIONS[name](gait)
    return breakdown


def fitness_function(gait: NDArray[np.float64]) -> float:
    """Fitness of one individual (a single scalar, higher is better).

    gait: (300, 8, 3) joint angles in radians. Out-of-range angles are valid
    input: they are scored (badly), never rejected or corrected.

    The theoretical upper bound is MAX_FITNESS; the result can be negative.
    """
    return float(BASE_FITNESS + sum(fitness_breakdown(gait).values()))
