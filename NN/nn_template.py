"""Skeleton for Part 2: predict 24 spider joint angles with a neural network."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from GA.GA import MAX_JOINT_ANGLES, MIN_JOINT_ANGLES, evaluate_fitness
from NN.NN import NeuralNetwork

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]

NUM_FRAMES = 300
NUM_OUTPUT_ANGLES = 24


@dataclass(frozen=True)
class FitnessScaler:
    """Store training-fitness limits so every data split uses the same scale."""

    minimum: float
    maximum: float

    def transform(self, values: FloatArray) -> FloatArray:
        """Scale fitness values to 0-1 for use as a network input."""
        span = self.maximum - self.minimum
        if span == 0:
            return np.zeros_like(values, dtype=np.float64)
        return (values - self.minimum) / span


@dataclass(frozen=True)
class Dataset:
    """Neural-network examples and the GA generation each example came from."""

    inputs: FloatArray
    targets: FloatArray
    generation_ids: IntArray


def select_best_gaits(generations: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Select the highest-fitness individual from every GA generation.

    Args:
        generations: Shape (generations, population, 300, 8, 3).

    Returns:
        best_gaits: Shape (generations, 300, 8, 3).
        best_fitness: Shape (generations,).
    """
    generations = np.asarray(generations, dtype=np.float64)
    if generations.ndim != 5 or generations.shape[2:] != (300, 8, 3):
        raise ValueError(
            "generations must have shape (generations, population, 300, 8, 3)"
        )

    fitness_by_generation = np.asarray(
        [evaluate_fitness(population) for population in generations]
    )
    best_indices = fitness_by_generation.argmax(axis=1)
    generation_indices = np.arange(len(generations))

    best_gaits = generations[generation_indices, best_indices].copy()
    best_fitness = fitness_by_generation[generation_indices, best_indices]
    return best_gaits, best_fitness


def split_generation_indices(
    number_of_generations: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[IntArray, IntArray, IntArray]:
    """Split complete generations into train, validation and test groups.

    Frames from one gait remain in one split, preventing nearly identical
    neighbouring frames from leaking into both training and testing.
    """
    if number_of_generations < 3:
        raise ValueError("at least three generations are required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    train_end = max(1, int(number_of_generations * train_fraction))
    validation_end = max(
        train_end + 1,
        int(number_of_generations * (train_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, number_of_generations - 1)

    indices = np.arange(number_of_generations)
    return (
        indices[:train_end],
        indices[train_end:validation_end],
        indices[validation_end:],
    )


def fit_fitness_scaler(training_fitness: FloatArray) -> FitnessScaler:
    """Fit fitness normalisation using training generations only."""
    training_fitness = np.asarray(training_fitness, dtype=np.float64)
    if training_fitness.size == 0 or not np.isfinite(training_fitness).all():
        raise ValueError("training fitness must contain finite values")
    return FitnessScaler(
        minimum=float(training_fitness.min()),
        maximum=float(training_fitness.max()),
    )


def normalise_joint_angles(gaits: FloatArray) -> FloatArray:
    """Scale legal joint angles to 0-1 for the 24 sigmoid outputs."""
    gaits = np.asarray(gaits, dtype=np.float64)
    ranges = MAX_JOINT_ANGLES - MIN_JOINT_ANGLES
    return (gaits - MIN_JOINT_ANGLES) / ranges


def denormalise_joint_angles(predictions: FloatArray) -> FloatArray:
    """Convert 24 network outputs from 0-1 back to legal angles in radians."""
    predictions = np.asarray(predictions, dtype=np.float64)
    minimums = MIN_JOINT_ANGLES.reshape(NUM_OUTPUT_ANGLES)
    ranges = (MAX_JOINT_ANGLES - MIN_JOINT_ANGLES).reshape(NUM_OUTPUT_ANGLES)
    clipped = np.clip(predictions, 0.0, 1.0)
    return minimums + clipped * ranges


def build_training_dataset(
    best_gaits: FloatArray,
    best_fitness: FloatArray,
    fitness_scaler: FitnessScaler,
) -> Dataset:
    """Turn every frame of every selected gait into one training example.

    Inputs:
        [normalised frame number, normalised gait fitness]

    Targets:
        24 normalised joint angles for that frame.
    """
    best_gaits = np.asarray(best_gaits, dtype=np.float64)
    best_fitness = np.asarray(best_fitness, dtype=np.float64)
    if best_gaits.ndim != 4 or best_gaits.shape[1:] != (300, 8, 3):
        raise ValueError("best_gaits must have shape (generations, 300, 8, 3)")
    if best_fitness.shape != (len(best_gaits),):
        raise ValueError("best_fitness must contain one value per gait")

    generation_count = len(best_gaits)
    frame_progress = np.tile(
        np.arange(NUM_FRAMES, dtype=np.float64) / (NUM_FRAMES - 1),
        generation_count,
    )
    scaled_fitness = fitness_scaler.transform(best_fitness)
    fitness_input: FloatArray = np.repeat(scaled_fitness, NUM_FRAMES)

    inputs = np.column_stack((frame_progress, fitness_input))
    targets = normalise_joint_angles(best_gaits).reshape(-1, NUM_OUTPUT_ANGLES)
    generation_ids: IntArray = np.repeat(np.arange(generation_count), NUM_FRAMES)
    return Dataset(inputs, targets, generation_ids)


def select_dataset_rows(dataset: Dataset, generation_ids: IntArray) -> Dataset:
    """Return only examples belonging to the requested GA generations."""
    mask = np.isin(dataset.generation_ids, generation_ids)
    return Dataset(
        inputs=dataset.inputs[mask],
        targets=dataset.targets[mask],
        generation_ids=dataset.generation_ids[mask],
    )


def build_network(seed: int = 42) -> NeuralNetwork:
    """Create the proposed 2 -> 128 -> 64 -> 24 network."""
    return NeuralNetwork(
        input_size=2,
        hidden_sizes=(128, 64),
        output_size=NUM_OUTPUT_ANGLES,
        seed=seed,
    )


def train_network(
    network: NeuralNetwork,
    training_data: Dataset,
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> list[float]:
    """Train with backpropagation and return loss for each epoch."""
    return network.train_nn(
        training_data.inputs,
        training_data.targets,
        epochs,
        learning_rate,
    )


def generate_gait(
    network: NeuralNetwork,
    desired_fitness: float,
    fitness_scaler: FitnessScaler,
) -> FloatArray:
    """Generate all 300 poses by requesting 24 angles for each frame."""
    scaled_fitness = float(
        fitness_scaler.transform(np.asarray([desired_fitness]))[0]
    )
    predicted_poses = []

    for frame in range(NUM_FRAMES):
        inputs: FloatArray = np.asarray(
            [frame / (NUM_FRAMES - 1), scaled_fitness],
            dtype=np.float64,
        )
        predicted_poses.append(network.feed_forward(inputs))

    angles = denormalise_joint_angles(np.asarray(predicted_poses))
    return angles.reshape(NUM_FRAMES, 8, 3)


def evaluate_generated_gait(
    predicted_gait: FloatArray,
    expected_gait: FloatArray,
) -> dict[str, float]:
    """Compare predicted angles with a GA target and calculate generated fitness."""
    predicted_gait = np.asarray(predicted_gait, dtype=np.float64)
    expected_gait = np.asarray(expected_gait, dtype=np.float64)
    if predicted_gait.shape != (300, 8, 3) or expected_gait.shape != (300, 8, 3):
        raise ValueError("both gaits must have shape (300, 8, 3)")

    error = predicted_gait - expected_gait
    within_limits = np.logical_and(
        predicted_gait >= MIN_JOINT_ANGLES,
        predicted_gait <= MAX_JOINT_ANGLES,
    )
    return {
        "angle_mae": float(np.mean(np.abs(error))),
        "angle_rmse": float(np.sqrt(np.mean(error**2))),
        "valid_angle_percentage": float(np.mean(within_limits) * 100),
        "generated_fitness": float(evaluate_fitness(predicted_gait[None])[0]),
    }


def plot_training_loss(losses: Sequence[float], output_path: Path) -> None:
    """Save the coursework-required loss-over-epochs visualisation."""
    figure, axis = plt.subplots()
    axis.plot(np.arange(1, len(losses) + 1), losses)
    axis.set(
        xlabel="Epoch",
        ylabel="Training loss",
        title="Neural-network training progress",
    )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_model(
    network: NeuralNetwork,
    fitness_scaler: FitnessScaler,
    output_path: Path,
) -> None:
    """Save architecture, parameters and normalisation required for testing."""
    model = {
        "layer_sizes": network.layer_sizes,
        "weights": network.weights,
        "biases": network.biases,
        "fitness_minimum": fitness_scaler.minimum,
        "fitness_maximum": fitness_scaler.maximum,
        "joint_minimums": MIN_JOINT_ANGLES,
        "joint_maximums": MAX_JOINT_ANGLES,
    }
    with output_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    """Show the intended orchestration; connect it to saved GA generations."""
    # 1. Load or run the GA to obtain `generations`.
    # 2. best_gaits, fitness = select_best_gaits(generations)
    # 3. Split generation indices before fitting the fitness scaler.
    # 4. Build the dataset and select train/validation/test rows.
    # 5. network = build_network()
    # 6. losses = train_network(network, training_data)
    # 7. Generate a gait and evaluate it against the held-out best GA gait.
    # 8. Plot losses, render predicted poses, and save the trained model.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
