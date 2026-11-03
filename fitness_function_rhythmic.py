"""EXPERIMENTAL alternative to the team's fitness_function.py.

Uses the reference at ALL 300 frames, rather than START/SPREAD/CLOSE
at frames 0/80/200/299. This rewards copying a designed rhythmic movement;
it does NOT measure Webots ground contacts, stability, or forward displacement.
"""
import numpy as np
from rhythmic_reference import STANDING, generate_reference

REFERENCE = generate_reference()


def losses(pop, target=REFERENCE):
    """Vectorized loss for populations shaped (n, 300, 8, 3). Lower is better."""
    mse = np.mean((pop - target[None, :, :, :]) ** 2, axis=(1, 2, 3))
    temporal = np.diff(pop, axis=1)
    ref_temporal = np.diff(target, axis=0)
    velocity_mismatch = np.mean(
        (temporal - ref_temporal[None, :, :, :]) ** 2, axis=(1, 2, 3)
    )
    jerk_proxy = np.mean(np.diff(pop, n=2, axis=1) ** 2, axis=(1, 2, 3))
    endpoints = np.mean(
        (pop[:, 0] - STANDING) ** 2 + (pop[:, -1] - STANDING) ** 2,
        axis=(1, 2),
    )
    return mse + 2.0 * velocity_mismatch + 0.5 * jerk_proxy + endpoints


def evaluate_fitness(pop):
    """Team-compatible fitness scores: one value per individual, HIGHER is better."""
    pop = np.asarray(pop, dtype=np.float64)
    if pop.ndim != 4 or pop.shape[1:] != REFERENCE.shape:
        raise ValueError(f"Population must have shape (n, {REFERENCE.shape})")
    return -losses(pop)


def fitness_function(gait):
    """A single individual's fitness, higher is better."""
    gait = np.asarray(gait, dtype=np.float64)
    if gait.shape != REFERENCE.shape:
        raise ValueError(f"Gait must have shape {REFERENCE.shape}")
    return float(-losses(gait[None])[0])
