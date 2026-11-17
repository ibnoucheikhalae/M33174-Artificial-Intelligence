"""Shared gait definitions used by evolution and fitness scoring."""

import numpy as np

NUM_FRAMES = 300

# Joint limits in radians. (hip, thigh, knee)
# DOESN'T limit crossover/mutation
# Used in: (1) fitness function (how far past) (2) Initialisation/Seeding and
# (3) Comparison of hip and knee error
RIGHT_MIN = [-1.57, -1.57, -2.0]
RIGHT_MAX = [1.57, 1.57, 1.0]

# Variables kept separate for readbility and comparison
LEFT_MIN = [-1.57, -1.57, -2.0]
LEFT_MAX = [1.57, 1.57, 1.0]


# Leg order: (index 0 = L1) L1, L2, L3, L4, R4, R3, R2, R1
MIN_JOINT_ANGLES = np.array([LEFT_MIN] * 4 + [RIGHT_MIN] * 4)
MAX_JOINT_ANGLES = np.array([LEFT_MAX] * 4 + [RIGHT_MAX] * 4)

#### POSE CYCLE (Step 1 of seeding)
# Keyframe poses, only used when scoring the generation and seeding the intiial population
# SPREAD: feet fanned around the body
# CLOSE: feet close together in pairs with the adjacent leg
# Note: hip movement is the "swing" of the leg, and knee movement is the "lift" thing
# Serve as target_pose_penalty within ff (looking at Frames 0, 150, 299)

SPREAD = np.array(
    [
        [-0.48, -1.3056, -0.8156],  # L1
        [-0.20, -1.3056, -0.8156],  # L2
        [0.20, -1.3056, -0.8156],  # L3
        [0.48, -1.3056, -0.8156],  # L4
        [-0.48, -1.3056, -0.8156],  # R4
        [-0.20, -1.3056, -0.8156],  # R3
        [0.20, -1.3056, -0.8156],  # R2
        [0.48, -1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

#!! todo: can we get the CLOSE to be more dramatic?
CLOSE = np.array(
    [
        [0.24, -1.3056, -0.8156],  # L1
        [0.10, -1.3056, -0.8156],  # L2
        [-0.10, -1.3056, -0.8156],  # L3
        [-0.24, -1.3056, -0.8156],  # L4
        [0.24, -1.3056, -0.8156],  # R4
        [0.10, -1.3056, -0.8156],  # R3
        [-0.10, -1.3056, -0.8156],  # R2
        [-0.24, -1.3056, -0.8156],  # R1
    ],
    dtype=float,
)

# Desired cycle: SPREAD 0, CLOSE 150, SPREAD 299

KEYFRAMES = np.array([0, 150, 299])
TARGET_POSES = np.stack([SPREAD, CLOSE, SPREAD])

#### STEP CYCLE (Step 2 of seeding)
# Two teams: A = L1 + L3, R2 + R4 // B = L2 + L4, R1 + R3
# When A's feet are up B's feet are down (each for half a cycle)
_STEP_PHASE_OFFSETS = np.array([0.0, np.pi] * 4)

# Ipsilateral couples (step together)
COUPLES = np.array([[0, 2], [1, 3], [7, 5], [6, 4]])

