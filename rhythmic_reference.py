"""Experimental sinusoidal gait reference and seeded initial population.

Place alongside GA.py. Uses the same constants and joint order as the team's GA.
The waveform is a hand-designed prior, NOT a physics-validated walking gait.
"""
import numpy as np
from GA import NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES

STANDING = np.array(
    [[0., -1.3056, 0.8156]] * 4 + [[0., 1.3056, -0.8156]] * 4,
    dtype=np.float64,
)
PHASE_OFFSET = np.array([0, np.pi, 0, np.pi, 0, np.pi, 0, np.pi])
HIP_DIRECTION = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
SIDE_SIGN = np.array([1] * 4 + [-1] * 4, dtype=float)


def easing_envelope():
    t = np.arange(NUM_FRAMES)
    r = np.clip(t / 18., 0., 1.)
    f = np.clip((NUM_FRAMES - 1 - t) / 18., 0., 1.)
    smooth = lambda x: x * x * (3 - 2 * x)
    return smooth(r) * smooth(f)


def generate_reference(phase_shifts=None, amps=None):
    """Reference 300x8x3 waveform with alternating sets of four legs."""
    if phase_shifts is None:
        phase_shifts = np.zeros(NUM_LEGS)
    if amps is None:
        amps = np.ones((NUM_LEGS, NUM_LEG_SEGS))
    phase = (2 * np.pi * np.arange(NUM_FRAMES)[:, None] / 75.
             + PHASE_OFFSET[None, :] + phase_shifts[None, :])
    env = easing_envelope()[:, None]
    hip = env * 0.24 * np.sin(phase) * HIP_DIRECTION[None, :] * amps[None, :, 0]
    lift = env * np.maximum(np.sin(phase), 0.) ** 2
    thigh = SIDE_SIGN[None, :] * 0.19 * lift * amps[None, :, 1]
    knee = -SIDE_SIGN[None, :] * 0.24 * lift * amps[None, :, 2]
    arr = np.broadcast_to(STANDING, (NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)).copy()
    arr[:, :, 0] += hip
    arr[:, :, 1] += thigh
    arr[:, :, 2] += knee
    return np.clip(arr, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)


def seeded_initial_population(n, rng):
    """Seed population near the rhythmic pattern; use one shared RNG throughout training."""
    population = np.empty((n, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS), dtype=np.float64)
    for i in range(n):
        phase_jitter = rng.normal(0., 0.40, NUM_LEGS)
        amplitude = rng.uniform(0.60, 1.40, (NUM_LEGS, NUM_LEG_SEGS))
        gait = generate_reference(phase_jitter, amplitude)
        knots = rng.normal(0, 0.06, (15, NUM_LEGS, NUM_LEG_SEGS))
        key_frames = np.linspace(0, NUM_FRAMES - 1, len(knots))
        for leg in range(NUM_LEGS):
            for joint in range(NUM_LEG_SEGS):
                gait[:, leg, joint] += (
                    np.interp(np.arange(NUM_FRAMES), key_frames, knots[:, leg, joint])
                    * easing_envelope()
                )
        gait[0] = STANDING
        gait[-1] = STANDING
        population[i] = np.clip(gait, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)
    return population
