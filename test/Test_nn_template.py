"""Regression tests for pose transitions, run splitting, and gait generation."""

import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from GA.GA import MAX_JOINT_ANGLES, MIN_JOINT_ANGLES
from NN import nn_template as nn
from NN.NN import NeuralNetwork


def example_gaits(count: int = 5) -> np.ndarray:
    """Return distinct legal gaits with hand-computable linear angle changes."""
    gaits = np.zeros((count, 300, 8, 3))
    for index in range(count):
        gaits[index, :, 0, 0] = index / 10 + np.arange(300) / 1000
    return gaits


class TestNNTemplate(unittest.TestCase):
    """Exercise the public pipeline with small deterministic training runs."""

    def test_build_training_dataset(self) -> None:
        """Pair adjacent poses without crossing the boundary between gaits."""
        gaits = example_gaits(2)
        original = gaits.copy()
        inputs, targets = nn.build_training_dataset(gaits)
        self.assertEqual(inputs.shape, (598, 25))
        self.assertEqual(targets.shape, (598, 24))
        self.assertAlmostEqual(inputs[0, 0], 0.5)
        self.assertAlmostEqual(targets[0, 0], (1.571 / 3.14))
        self.assertAlmostEqual(inputs[298, -1], 298 / 299)
        self.assertEqual(inputs[299, -1], 0)
        self.assertAlmostEqual(inputs[299, 0], 1.67 / 3.14)
        self.assertAlmostEqual(targets[298, 0], 1.869 / 3.14)
        np.testing.assert_array_equal(original, gaits)

    def test_split_gaits(self) -> None:
        """Keep every individual from a run in the same reproducible split."""
        # Three individuals per run: splitting individual gaits would leak ancestry.
        run_ids = np.repeat([f"run_{index}.npz" for index in range(10)], 3)
        training, testing = nn.split_gaits(run_ids, seed=7)
        self.assertEqual(len(training), 24)
        self.assertEqual(len(testing), 6)
        self.assertFalse(set(run_ids[training]) & set(run_ids[testing]))
        other_training, other_testing = nn.split_gaits(run_ids, seed=7)
        np.testing.assert_array_equal(training, other_training)
        np.testing.assert_array_equal(testing, other_testing)
        with self.assertRaises(ValueError):
            nn.split_gaits(np.repeat("one_run.npz", 100))
        training, testing = nn.split_gaits(np.array(["a", "b"]))
        self.assertEqual(len(training), 1)
        self.assertEqual(len(testing), 1)

    def test_load_gaits(self) -> None:
        """Read all individuals, retain run IDs, deduplicate, and validate once."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(ValueError):
                nn.load_gaits(folder)
            gaits = example_gaits(4)
            np.savez_compressed(folder / "run_0.npz", gaits=gaits[:2])
            np.savez_compressed(folder / "run_1.npz", gaits=gaits[1:])
            loaded, run_ids = nn.load_gaits(folder)
            np.testing.assert_array_equal(loaded, gaits)
            np.testing.assert_array_equal(
                run_ids, ["run_0.npz", "run_0.npz", "run_1.npz", "run_1.npz"]
            )
            for invalid in (
                np.zeros((299, 8, 3)),
                np.full((1, 300, 8, 3), np.nan),
                np.full((1, 300, 8, 3), 10.0),
                np.empty((0, 300, 8, 3)),
            ):
                np.savez_compressed(folder / "bad.npz", gaits=invalid)
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    nn.load_gaits(folder)

    def test_generate_gait(self) -> None:
        """Use each prediction as the next input and advance the current frame."""
        # A real network with known weights makes feedback and timing observable.
        network = NeuralNetwork(25, (), 24, seed=1)
        network.weights[0][:] = 0
        network.biases[0][:] = 0
        network.weights[0][0, 0] = 1
        network.weights[0][24, 0] = 1
        pose = MIN_JOINT_ANGLES.copy()
        gait = nn.generate_gait(network, pose)
        self.assertEqual(gait.shape, (300, 8, 3))
        np.testing.assert_array_equal(gait[0], pose)
        self.assertAlmostEqual(gait[1, 0, 0], 0)
        second_prediction = 1 / (1 + np.exp(-(0.5 + 1 / 299)))
        self.assertAlmostEqual(gait[2, 0, 0], -1.57 + 3.14 * second_prediction)
        self.assertTrue((gait >= MIN_JOINT_ANGLES).all())
        self.assertTrue((gait <= MAX_JOINT_ANGLES).all())
        np.testing.assert_array_equal(pose, MIN_JOINT_ANGLES)
        for invalid in (np.zeros(24), np.full((8, 3), np.inf), np.full((8, 3), 10)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                nn.generate_gait(network, invalid)

    def test_train_network(self) -> None:
        """Train a 25-input network and reduce loss on a small legal dataset."""
        inputs, targets = nn.build_training_dataset(example_gaits(1))
        network = nn.build_network(seed=3)
        self.assertEqual(network.layer_sizes, (25, 40, 24))
        losses = nn.train_network(network, inputs[:10], targets[:10], 20, 0.1)
        self.assertEqual(len(losses), 20)
        self.assertLess(losses[-1], losses[0])

    def test_training_progress_callback(self) -> None:
        """Report each completed epoch without changing training or printing internally."""
        inputs, targets = nn.build_training_dataset(example_gaits(1))
        observed = []

        def record_epoch(epoch: int, loss: float) -> None:
            """Collect progress for comparison with the returned training history."""
            observed.append((epoch, loss))

        output = io.StringIO()
        network = nn.build_network(seed=3)
        with redirect_stdout(output):
            losses = nn.train_network(
                network, inputs[:5], targets[:5], 3, 0.1, on_epoch=record_epoch
            )
        self.assertEqual([epoch for epoch, _ in observed], [1, 2, 3])
        self.assertEqual([loss for _, loss in observed], losses)
        self.assertEqual(output.getvalue(), "")
        comparison = nn.build_network(seed=3)
        self.assertEqual(nn.train_network(comparison, inputs[:5], targets[:5], 3, 0.1), losses)

    def test_evaluate_network(self) -> None:
        """Measure next-pose prediction error against a persistence baseline."""
        network = NeuralNetwork(25, (), 24)
        network.weights[0][:] = 0
        network.biases[0][:] = 0
        gaits = np.broadcast_to((MIN_JOINT_ANGLES + MAX_JOINT_ANGLES) / 2, (1, 300, 8, 3)).copy()
        metrics = nn.evaluate_network(network, gaits)
        self.assertEqual(metrics["next_pose_mse"], 0)
        self.assertEqual(metrics["persistence_mse"], 0)
        gaits[0, 1, 0, 0] += 0.314
        metrics = nn.evaluate_network(network, gaits)
        self.assertAlmostEqual(metrics["next_pose_mse"], 0.01 / (299 * 24))
        self.assertAlmostEqual(metrics["persistence_mse"], 0.02 / (299 * 24))

    def test_evaluate_generated_gait(self) -> None:
        """Report rollout errors separately from the supplied seed pose."""
        expected = example_gaits(1)[0]
        predicted = expected.copy()
        predicted[1:] += 0.1
        metrics = nn.evaluate_generated_gait(predicted, expected)
        self.assertAlmostEqual(metrics["angle_mae"], 0.1)
        self.assertAlmostEqual(metrics["angle_rmse"], 0.1)
        self.assertEqual(metrics["valid_angle_percentage"], 100)
        self.assertTrue(np.isfinite(metrics["generated_fitness"]))
        self.assertIn("mean_absolute_step", metrics)
        self.assertIn("mean_absolute_acceleration", metrics)

    def test_save_and_load_model(self) -> None:
        """Restore architecture, parameters, and identical next-pose predictions."""
        network = nn.build_network(seed=2)
        inputs, _ = nn.build_training_dataset(example_gaits(1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            nn.save_model(network, path)
            restored = nn.load_model(path)
            self.assertEqual(restored.layer_sizes, (25, 40, 24))
            np.testing.assert_array_equal(
                network.feed_forward(inputs[0]), restored.feed_forward(inputs[0])
            )
            with path.open("rb") as file:
                model = pickle.load(file)
            incompatible: list[dict[str, object]] = [
                {"format": "old_fitness_model"},
                {"num_frames": 299},
                {"joint_minimums": np.zeros((8, 3))},
                {"joint_maximums": np.zeros((8, 3))},
                {"layer_sizes": (2, 40, 24)},
                {"weights": []},
                {"biases": [np.zeros(1), np.zeros(24)]},
                {"weights": [np.full((25, 40), np.nan), np.zeros((40, 24))]},
            ]
            for change in incompatible:
                with self.subTest(field=next(iter(change))):
                    with path.open("wb") as file:
                        pickle.dump({**model, **change}, file)
                    with self.assertRaises(ValueError):
                        nn.load_model(path)

    def test_main(self) -> None:
        """Run training, held-out generation, and artifact export end to end."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "runs"
            data.mkdir()
            for index, gait in enumerate(example_gaits(2)):
                np.savez_compressed(data / f"run_{index}.npz", gaits=gait[None])
            output = folder / "results"
            progress = io.StringIO()
            with redirect_stdout(progress):
                self.assertEqual(
                    nn.main([str(data), "--output-dir", str(output), "--epochs", "1"]), 0
                )
            self.assertIn("Epoch 1/1", progress.getvalue())
            self.assertIn("MSE", progress.getvalue())
            self.assertTrue((output / "training_loss.png").is_file())
            self.assertTrue((output / "model.pkl").is_file())
            generated = np.load(output / "generated_gaits.npy")
            self.assertEqual(generated.shape, (1, 300, 8, 3))
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["training_gaits"], 1)
            self.assertEqual(metrics["testing_gaits"], 1)
            self.assertIn("next_pose_mse", metrics)
            self.assertIn("persistence_fitness", metrics["rollouts"][0])
            source = metrics["testing_sources"][0]
            with np.load(data / source, allow_pickle=False) as archive:
                reference = archive["gaits"][0]
            np.testing.assert_array_equal(generated[0, 0], reference[0])


if __name__ == "__main__":
    unittest.main()
