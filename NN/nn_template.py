"""Train a next-pose spider network using automatically exported GA gaits.

From the repository root::

    python -m GA.GA
    python -m NN.nn_template --output-dir nn_results --epochs 100
    python -m GA.render_gait nn_results/generated_gaits.npy

Training prints mean training MSE after each completed epoch. The renderer
chooses the generated gait with the highest current GA fitness.

The GA script and configuration experiment save distinct individuals with fitness strictly
above GA.FITNESS_SAVE_THRESHOLD into NN/data/run_<id>.npz after evolution ends.
An archive contains gaits (N, 300, 8, 3) in radians, fitness, and source generations.
Run the GA multiple times to collect independent training and testing runs.

The loader validates data once and removes exact duplicates across archives.
The NN splits complete runs 80:20, keeping all descendants of a run together.
The ratio is by run count, so gait counts can differ from 80:20. Near-duplicates
and runs with shared initial populations still require care in experiment design.

Each gait supplies 299 transitions. Inputs are the 24 current joint angles in
leg-major order, followed by current frame / 299; targets are the 24 next angles.
The default network is 25 -> 40 -> 24 with sigmoid activations, MSE loss, and
backpropagation implemented by NN.NeuralNetwork.

Why scale angles?
-----------------
Sigmoid outputs lie in (0, 1), so they cannot directly predict negative radians
or radians above 1. Fixed joint limits map valid radians reversibly into [0, 1]:
    scaled = (angle - minimum) / (maximum - minimum)
An out-of-range angle scales outside [0, 1]; scaling does not silently fix it.
Such data is rejected once when loaded, with a clear error. If physical limits
change, update the GA limits and retrain. Files and generated gaits stay in
radians; only network inputs/targets use this internal representation.

Evaluation measures held-out next-pose error and complete sequences generated
by repeatedly feeding predictions back in, starting with a reference's first
pose. Persistence baselines copy the current pose for one-step predictions and
hold the starting pose for complete sequences. Outputs are model.pkl,
training_loss.png, metrics.json, and generated_gaits.npy. Metrics include source
archive names in generated-gait order. A new seed pose does not guarantee novel
or effective motion. Reserve validation runs before tuning hyperparameters.

Validation belongs at file loading, network construction, and externally supplied
starting poses. Dataset building and training assume these boundaries have
already checked their inputs. Only model files use pickle; load trusted models.
"""

import argparse
import json
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from GA.GA import GAIT_DATA_DIRECTORY, evaluate_fitness
from GA.gait_constants import MAX_JOINT_ANGLES, MIN_JOINT_ANGLES, NUM_FRAMES
from GA.kinematics import NUM_LEG_SEGS, NUM_LEGS
from NN.NN import NeuralNetwork

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
StringArray = NDArray[np.str_]
POSE_SHAPE = (NUM_LEGS, NUM_LEG_SEGS)
GAIT_SHAPE = (NUM_FRAMES, *POSE_SHAPE)
NUM_ANGLES = NUM_LEGS * NUM_LEG_SEGS
NUM_INPUTS = NUM_ANGLES + 1  # All joint angles, plus the current frame.
NUM_TRANSITIONS = NUM_FRAMES - 1
DEFAULT_HIDDEN_SIZES = (40,)


def validate_gaits(gaits: FloatArray) -> FloatArray:
    """Validate externally loaded gaits once: shape, finite values, and joint limits.

    Args:
        gaits: Array of shape (number_of_gaits, 300, 8, 3), in radians.

    Returns:
        The validated array. No input values are changed or clipped.

    Raises:
        ValueError: Shape, finiteness, or joint limits are invalid.
    """
    gaits = np.asarray(gaits, dtype=np.float64)
    if gaits.ndim != 4 or gaits.shape[1:] != GAIT_SHAPE or len(gaits) == 0:
        raise ValueError("gaits must have nonempty shape (number_of_gaits, 300, 8, 3)")
    if not np.isfinite(gaits).all():
        raise ValueError("gaits must contain finite angles")
    if np.any(gaits < MIN_JOINT_ANGLES) or np.any(gaits > MAX_JOINT_ANGLES):
        raise ValueError("gaits contain angles outside the permitted GA joint limits")
    return gaits


def load_gaits(directory: Path = GAIT_DATA_DIRECTORY) -> tuple[FloatArray, StringArray]:
    """Load automatically exported run archives and validate angles on entry.

    Returns gaits (N, 300, 8, 3) in radians and run_ids (N,), containing the
    source archive name for each gait. Exact duplicates across archives are
    removed, retaining the first occurrence and its source run. Empty or
    malformed angle arrays raise ValueError. NumPy archives do not use pickle.
    """
    gait_batches = []
    run_ids = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            run_gaits = validate_gaits(archive["gaits"])
        gait_batches.append(run_gaits)
        run_ids.extend([path.name] * len(run_gaits))
    if not gait_batches:
        raise ValueError(f"no GA training archives (*.npz) found in {directory}")

    # concatenate joins the runs along the gait axis, retaining every individual.
    gaits = np.concatenate(gait_batches)
    # unique compares whole gaits along axis 0; keep their first source run.
    _, first_occurrences = np.unique(gaits, axis=0, return_index=True)
    first_occurrences.sort()  # Preserve original file/gait order after deduplication.
    return gaits[first_occurrences], np.asarray(run_ids)[first_occurrences]


def split_gaits(
    run_ids: StringArray, train_fraction: float = 0.8, seed: int = 42
) -> tuple[IntArray, IntArray]:
    """Return training/test gait indices, splitting complete runs (default 80:20).

    run_ids has one source archive name per loaded gait. Shuffle distinct run
    names reproducibly, then keep all individuals of a run in the same split.
    Round the training run count down, keeping at least one run in each set.
    Requires at least two remaining runs after loading and deduplication.
    """
    runs = np.unique(run_ids)  # Each run name appears once, regardless of its size.
    if len(runs) < 2:
        raise ValueError("at least two independent GA runs with distinct gaits are required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    train_count = max(1, min(int(len(runs) * train_fraction), len(runs) - 1))
    # isin creates a Boolean mask: True for each gait from a selected training run.
    training = np.isin(run_ids, runs[:train_count])
    # flatnonzero returns the positions where a mask is True, for array indexing.
    return np.flatnonzero(training), np.flatnonzero(~training)


def normalise_joint_angles(angles: FloatArray) -> FloatArray:
    """Scale radians to [0, 1] using GA limits, preserving (..., 8, 3) shape.

    Callers validate the angles first; this arithmetic does not clip them.
    The fixed limits require no fitting on training or test data.
    """
    return (np.asarray(angles, dtype=np.float64) - MIN_JOINT_ANGLES) / (
        MAX_JOINT_ANGLES - MIN_JOINT_ANGLES
    )


def denormalise_joint_angles(predictions: FloatArray) -> FloatArray:
    """Map (..., 24) sigmoid predictions to flattened legal angles in radians.

    Clipping guards the joint limits; it does not ensure stability or movement.
    """
    minimums = MIN_JOINT_ANGLES.reshape(NUM_ANGLES)
    ranges = (MAX_JOINT_ANGLES - MIN_JOINT_ANGLES).reshape(NUM_ANGLES)
    # clip bounds numerical predictions; it is not applied to loaded training angles.
    return minimums + np.clip(predictions, 0.0, 1.0) * ranges


def build_training_dataset(gaits: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Pair poses at t with poses at t+1, keeping transitions within each gait.

    Args:
        gaits: Legal radians of shape (number_of_gaits, 300, 8, 3).

    Returns:
        inputs: Shape (number_of_gaits * 299, 25). Each row contains 24
            normalised current angles, followed by current frame / 299.
        targets: Shape (number_of_gaits * 299, 24), normalised next angles.

    Rows are ordered by gait then frame. Joint order is unchanged: the three
    joints of leg 0, then leg 1, etc. There is no last-to-first transition.
    Split whole gaits BEFORE calling this function separately for each set.
    """
    # reshape flattens each pose's legs/joints into one vector of 24 angles.
    poses = normalise_joint_angles(gaits).reshape(len(gaits), NUM_FRAMES, NUM_ANGLES)
    current_poses = poses[:, :-1].reshape(-1, NUM_ANGLES)
    targets = poses[:, 1:].reshape(-1, NUM_ANGLES)
    # arange gives frame indices 0..298; tile repeats that sequence for each gait.
    frames = np.tile(np.arange(NUM_TRANSITIONS) / NUM_TRANSITIONS, len(gaits))
    # column_stack appends the frame column to each current-pose input row.
    inputs = np.column_stack((current_poses, frames))
    return inputs, targets


def build_network(
    seed: int = 42, hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES
) -> NeuralNetwork:
    """Create a sigmoid network with 25 inputs, chosen hidden widths, and 24 outputs.

    Defaults to one 40-neuron hidden layer. Use centred Xavier-uniform weights
    so the larger layers do not start with saturated sigmoid outputs. Biases
    remain zero. This only changes initialisation for this gait model.
    """
    network = NeuralNetwork(
        input_size=NUM_INPUTS, hidden_sizes=hidden_sizes, output_size=NUM_ANGLES, seed=seed
    )
    rng = np.random.default_rng(seed)
    for index, weights in enumerate(network.weights):
        source_size, destination_size = weights.shape
        # Xavier's bound is sqrt(6 / (number of inputs + number of outputs)).
        limit = np.sqrt(6.0 / (source_size + destination_size))
        network.weights[index] = rng.uniform(-limit, limit, weights.shape)
    return network


def train_network(
    network: NeuralNetwork,
    inputs: FloatArray,
    targets: FloatArray,
    epochs: int = 100,
    learning_rate: float = 0.1,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train on transition rows and return mean training MSE for each epoch.

    Inputs have shape (N, 25) and targets (N, 24), with N > 0. The underlying
    train_nn updates after each example, in supplied order. No test rows are
    used for updates. Rows are produced by build_training_dataset, and training
    options are checked once by main. Tuning requires separate validation data.
    on_epoch optionally receives the completed epoch (starting at 1) and its
    mean loss, letting the caller report progress without logging here.
    """
    return network.train_nn(inputs, targets, epochs, learning_rate, on_epoch=on_epoch)


def generate_gait(network: NeuralNetwork, initial_pose: FloatArray) -> FloatArray:
    """Generate 300 poses in radians, feeding each predicted pose back as input.

    Args:
        network: A trained 25-input, 24-output next-pose network.
        initial_pose: Legal starting angles of shape (8, 3), in radians.

    Returns:
        Shape (300, 8, 3). Frame 0 is the supplied pose; the remaining 299
        frames are predictions. The final pose is not forced to match the seed.

    No reference poses are supplied after frame 0. Errors may accumulate, and
    identical initial poses produce identical sequences for the same network.
    """
    initial_pose = np.asarray(initial_pose, dtype=np.float64)
    if initial_pose.shape != POSE_SHAPE:
        raise ValueError("initial_pose must have shape (8, 3)")
    if not np.isfinite(initial_pose).all():
        raise ValueError("initial_pose must contain finite angles")
    if np.any(initial_pose < MIN_JOINT_ANGLES) or np.any(initial_pose > MAX_JOINT_ANGLES):
        raise ValueError("initial_pose must be within the GA joint limits")
    gait = np.empty(GAIT_SHAPE, dtype=np.float64)
    gait[0] = initial_pose

    for frame in range(NUM_TRANSITIONS):
        # append adds the current frame after the flattened current-pose angles.
        inputs = np.append(
            normalise_joint_angles(gait[frame]).reshape(NUM_ANGLES), frame / NUM_TRANSITIONS
        )
        predictions = network.feed_forward(inputs)
        if not np.isfinite(predictions).all():
            raise ValueError("network produced non-finite predictions")
        gait[frame + 1] = denormalise_joint_angles(predictions).reshape(POSE_SHAPE)
    return gait


def evaluate_network(network: NeuralNetwork, gaits: FloatArray) -> dict[str, float]:
    """Measure next-pose MSE on held-out GA poses, without updating parameters.

    Reports normalised-angle MSE for the network and for persistence (copying
    the input pose). These are single-step predictions with true input poses;
    use evaluate_generated_gait to assess accumulated generation errors.
    """
    inputs, targets = build_training_dataset(gaits)
    predictions = np.asarray([network.feed_forward(example) for example in inputs])
    return {
        "next_pose_mse": float(np.mean((predictions - targets) ** 2)),
        "persistence_mse": float(np.mean((inputs[:, :NUM_ANGLES] - targets) ** 2)),
    }


def evaluate_generated_gait(
    predicted_gait: FloatArray, expected_gait: FloatArray
) -> dict[str, float]:
    """Measure complete-sequence quality against aligned held-out reference poses.

    MAE/RMSE are radians over frames 1..299, excluding the supplied seed.
    Joint-limit compliance and actual GA fitness use the complete gait.
    Mean absolute step and acceleration measure first and second differences
    (radians/frame and radians/frame squared). Small values can also indicate
    a static gait; interpret them together with fitness and rendered movement.
    Out-of-limit/non-finite gaits are rejected by the GA fitness calculation.
    """
    error = predicted_gait[1:] - expected_gait[1:]
    within_limits = (predicted_gait >= MIN_JOINT_ANGLES) & (predicted_gait <= MAX_JOINT_ANGLES)
    # diff computes consecutive-frame differences; n=2 gives changes in those steps.
    return {
        "angle_mae": float(np.mean(np.abs(error))),
        "angle_rmse": float(np.sqrt(np.mean(error**2))),
        "valid_angle_percentage": float(np.mean(within_limits) * 100),
        "generated_fitness": float(evaluate_fitness(predicted_gait[None])[0]),
        "mean_absolute_step": float(np.mean(np.abs(np.diff(predicted_gait, axis=0)))),
        "mean_absolute_acceleration": float(np.mean(np.abs(np.diff(predicted_gait, n=2, axis=0)))),
    }


def plot_training_loss(losses: Sequence[float], output_path: Path) -> None:
    """Save training MSE against epoch number to an existing output directory."""
    figure, axis = plt.subplots()
    axis.plot(np.arange(1, len(losses) + 1), losses)
    axis.set(xlabel="Epoch", ylabel="Mean squared error", title="Next-pose training loss")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_model(network: NeuralNetwork, output_path: Path) -> None:
    """Pickle parameters and preprocessing metadata for load_model to restore.

    Stores joint limits, frame count, and format version alongside architecture,
    weights, and biases. The destination directory must already exist.
    """
    model = {
        "format": "next_pose_v1",
        "layer_sizes": network.layer_sizes,
        "weights": network.weights,
        "biases": network.biases,
        "joint_minimums": MIN_JOINT_ANGLES,
        "joint_maximums": MAX_JOINT_ANGLES,
        "num_frames": NUM_FRAMES,
    }
    with output_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(input_path: Path) -> NeuralNetwork:
    """Restore a trusted next-pose model, rejecting incompatible preprocessing.

    Old frame/fitness models are incompatible. Pickle may execute Python code,
    so only open model files you trust.
    """
    with input_path.open("rb") as file:
        model = pickle.load(file)
    if model.get("format") != "next_pose_v1" or model["num_frames"] != NUM_FRAMES:
        raise ValueError("expected a next_pose_v1 model with the current frame count")
    for key, limits in (("joint_minimums", MIN_JOINT_ANGLES), ("joint_maximums", MAX_JOINT_ANGLES)):
        if not np.array_equal(model[key], limits):
            raise ValueError("model joint limits differ from the current GA limits")
    layer_sizes = model["layer_sizes"]
    if layer_sizes[0] != NUM_INPUTS or layer_sizes[-1] != NUM_ANGLES:
        raise ValueError("model must have 25 inputs and 24 outputs")
    network = build_network(hidden_sizes=tuple(layer_sizes[1:-1]))
    for key in ("weights", "biases"):
        parameters = model[key]
        expected = getattr(network, key)
        if len(parameters) != len(expected):
            raise ValueError(f"incorrect number of model {key}")
        for values, template in zip(parameters, expected):
            if values.shape != template.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid model {key}")
        setattr(network, key, [values.copy() for values in parameters])
    return network


def main(argv: Sequence[str] | None = None) -> int:
    """Load saved runs, train, evaluate held-out gaits, and save results.

    argv defaults to command-line arguments. Run as python -m NN.nn_template
    from the repository root; use --help for paths and training options.
    Each output gait starts from the corresponding held-out reference's seed.
    metrics.json records per-gait rollout and baseline results in array order.
    testing_sources lists each generated gait's source archive in array order.
    The optional gait_directory defaults to the GA's automatic export directory.
    """
    parser = argparse.ArgumentParser(
        description="Train a next-pose network on automatically exported GA gaits."
    )
    parser.add_argument("gait_directory", type=Path, nargs="?", default=GAIT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=Path("nn_results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive and finite")

    # 1. Split independent runs before constructing any frame-level examples.
    gaits, run_ids = load_gaits(args.gait_directory)
    training_indices, testing_indices = split_gaits(run_ids, seed=args.seed)
    training_gaits = gaits[training_indices]
    testing_gaits = gaits[testing_indices]
    inputs, targets = build_training_dataset(training_gaits)

    # 2. Train only on training rows; keep testing rows out of all updates.
    network = build_network(args.seed, tuple(args.hidden_sizes))
    print(
        f"Training on {len(training_gaits)} gaits ({len(inputs)} transitions per epoch); "
        f"holding out {len(testing_gaits)} gaits. Starting {args.epochs} epochs.",
        flush=True,
    )

    def report_epoch(epoch: int, mean_loss: float) -> None:
        """Print each completed epoch's mean training MSE immediately."""
        print(f"Epoch {epoch}/{args.epochs}: training MSE={mean_loss:.8f}", flush=True)

    losses = train_network(
        network, inputs, targets, args.epochs, args.learning_rate, on_epoch=report_epoch
    )
    print("Evaluating held-out transitions and generating gaits...", flush=True)
    metrics = evaluate_network(network, testing_gaits)
    metrics["training_gaits"] = len(training_gaits)
    metrics["testing_gaits"] = len(testing_gaits)
    rollouts: list[dict[str, float]] = []
    report = {
        **metrics,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "layer_sizes": network.layer_sizes,
        "rollouts": rollouts,
        "testing_sources": run_ids[testing_indices].tolist(),
    }

    # 3. Generate entire held-out sequences using only their starting poses.
    generated_gaits = []
    for reference in testing_gaits:
        generated = generate_gait(network, reference[0])
        result = evaluate_generated_gait(generated, reference)
        # repeat copies the initial pose along the frame axis to make a static gait.
        persistence = np.repeat(reference[:1], NUM_FRAMES, axis=0)
        baseline = evaluate_generated_gait(persistence, reference)
        result["persistence_fitness"] = baseline["generated_fitness"]
        result["persistence_angle_rmse"] = baseline["angle_rmse"]
        result["reference_fitness"] = float(evaluate_fitness(reference[None])[0])
        rollouts.append(result)
        generated_gaits.append(generated)

    # 4. Save results for later generation, plotting, and inspection.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_model(network, args.output_dir / "model.pkl")
    plot_training_loss(losses, args.output_dir / "training_loss.png")
    np.save(args.output_dir / "generated_gaits.npy", np.asarray(generated_gaits))
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
