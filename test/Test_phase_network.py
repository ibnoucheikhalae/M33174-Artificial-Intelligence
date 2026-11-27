"""Exercise the single phase workflow through its existing public functions."""

import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from GA.gait_constants import MAX_JOINT_ANGLES, MIN_JOINT_ANGLES
from NN.NN import (
    NeuralNetwork,
    create_phase_inputs,
    generate_phase_gait,
    load_network,
    train_network,
)


class TestPhaseNetwork(unittest.TestCase):
    def test_create_phase_inputs(self):
        inputs = create_phase_inputs(300)
        self.assertEqual(inputs.shape, (300, 5))
        t = np.linspace(0, 1, 300)
        for column, expected in enumerate(
            (
                t,
                np.sin(2 * np.pi * t),
                np.cos(2 * np.pi * t),
                np.sin(4 * np.pi * t),
                np.cos(4 * np.pi * t),
            )
        ):
            np.testing.assert_allclose(inputs[:, column], expected)
        np.testing.assert_allclose(inputs[0], [0, 0, 1, 0, 1])
        np.testing.assert_allclose(inputs[-1], [1, 0, 1, 0, 1], atol=1e-14)

    def test_generate_phase_gait(self):
        network = NeuralNetwork(5, (), 24, seed=1, output_activation="linear")
        network.weights[0][:] = 0
        network.biases[0][:] = 0
        network.weights[0][1, 0] = 2
        inputs = create_phase_inputs(300)
        predictions = np.array([network.feed_forward(x) for x in inputs])
        expected = predictions.reshape(300, 8, 3)
        np.testing.assert_array_equal(generate_phase_gait(network), expected)
        # Each frame uses its own phase, including frame zero; there is no seed pose.
        self.assertGreater(np.ptp(expected[:, 0, 0]), 0.1)
        self.assertGreater(expected.max(), 1.0)
        network.weights[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            generate_phase_gait(network)

    def test_train_phase_network(self):
        """Train on GA poses at the same frame, save loss, and reload identically."""
        gait = np.zeros((300, 8, 3))
        gait[:, 0, 0] = np.arange(300) / 1000
        original_train = NeuralNetwork.train_nn
        seen_targets = []

        def record_training(network, inputs, targets, **kwargs):
            np.testing.assert_array_equal(inputs, create_phase_inputs(300))
            seen_targets.append(targets.copy())
            return original_train(network, inputs, targets, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            gait_path, model_path = folder / "data", folder / "model.pkl"
            gait_path.mkdir()
            gait[:, 0, 1] = -3.0  # Raw targets may exceed the GA's assumed joint range.
            for stored in (gait[None],):
                np.savez(gait_path / "run_test.npz", gaits=stored)
                with (
                    patch.object(NeuralNetwork, "train_nn", record_training),
                    redirect_stdout(io.StringIO()),
                ):
                    network = train_network(gait_path, model_path, epochs=1)
                    loaded = load_network(model_path)
                self.assertEqual(network.layer_sizes, (5, 64, 24))
                np.testing.assert_array_equal(
                    generate_phase_gait(network), generate_phase_gait(loaded)
                )
                self.assertTrue(
                    (folder / "training_loss.png").read_bytes().startswith(b"\x89PNG")
                )
            expected = gait.reshape(300, 24)
            for targets in seen_targets:
                np.testing.assert_array_equal(targets, expected)
            for invalid in (
                gait[:299],
                np.full_like(gait, np.nan),
            ):
                np.savez(gait_path / "run_test.npz", gaits=invalid[None])
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    train_network(gait_path, model_path, epochs=1)

    def test_train_phase_network_from_ga_archives(self):
        """Use every saved gait, with time restarting for each individual."""
        from GA.GA import save_training_gaits

        gaits = np.zeros((3, 300, 8, 3))
        gaits[1, :, 0, 0] = 0.1
        gaits[2, :, 0, 0] = 0.2
        original_train = NeuralNetwork.train_nn
        seen = []

        def record_training(network, inputs, targets, **kwargs):
            seen.append((inputs.copy(), targets.copy()))
            return original_train(network, inputs, targets, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "data"
            model = folder / "output/model.pkl"
            with self.assertRaisesRegex(ValueError, "run_.*npz"):
                data.mkdir()
                train_network(data, model, epochs=1)
            save_training_gaits(gaits[:2], np.ones(2), np.zeros(2), data)
            save_training_gaits(gaits[2:], np.ones(1), np.zeros(1), data)
            with patch.object(NeuralNetwork, "train_nn", record_training):
                network = train_network(data, model, epochs=1)
            self.assertTrue(model.exists())
            np.testing.assert_array_equal(
                generate_phase_gait(network), generate_phase_gait(load_network(model))
            )
            inputs, targets = seen[0]
            np.testing.assert_array_equal(inputs, np.tile(create_phase_inputs(300), (3, 1)))
            expected = gaits.reshape(3, 300, 24)
            # Archive filenames are random, so compare individuals without assuming order.
            self.assertEqual(targets.shape, (900, 24))
            self.assertEqual(
                sorted(block.tobytes() for block in targets.reshape(3, 300, 24)),
                sorted(block.tobytes() for block in expected),
            )

    def test_load_phase_network(self):
        """Preserve the original compatibility checks for saved models."""
        network = NeuralNetwork(5, (), 24, seed=1, output_activation="linear")
        saved = {
            "format": "phase_pose_linear_v1",
            "layer_sizes": network.layer_sizes,
            "weights": network.weights,
            "biases": network.biases,
            "joint_minimums": MIN_JOINT_ANGLES,
            "joint_maximums": MAX_JOINT_ANGLES,
            "num_frames": 300,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            for change in (
                {"layer_sizes": (25, 40, 24)},
                {"num_frames": 299},
                {"format": "phase_pose_v1"},
                {"weights": []},
                {"biases": [np.full(24, np.nan)]},
            ):
                with path.open("wb") as file:
                    pickle.dump({**saved, **change}, file)
                with self.subTest(change=change.keys()), self.assertRaises(ValueError):
                    load_network(path)


class TestLinearOutput(unittest.TestCase):
    def test_unbounded_outputs_and_sigmoid_default(self):
        linear = NeuralNetwork(1, (), 2, seed=0, output_activation="linear")
        linear.weights[0][:] = [[-4, 4]]
        np.testing.assert_array_equal(linear.feed_forward(np.array([1.])), [-4, 4])
        sigmoid = NeuralNetwork(1, (), 2, seed=0)
        sigmoid.weights[0][:] = linear.weights[0]
        self.assertTrue(((sigmoid.feed_forward(np.array([1.])) > 0) &
                         (sigmoid.feed_forward(np.array([1.])) < 1)).all())

    def test_linear_output_gradients(self):
        network = NeuralNetwork(2, (3,), 2, seed=0, output_activation="linear")
        x, target = np.array([-.3, .7]), np.array([-3., 4.])
        network.feed_forward(x)
        network.back_propagation(target)
        for parameters, gradients in ((network.weights, network.weight_gradients),
                                       (network.biases, network.bias_gradients)):
            for parameter, gradient in zip(parameters, gradients):
                expected = gradient.copy()
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + 1e-6
                    plus = network.loss(target, network.feed_forward(x))
                    parameter[index] = original - 1e-6
                    minus = network.loss(target, network.feed_forward(x))
                    parameter[index] = original
                    self.assertAlmostEqual(expected[index], (plus-minus)/2e-6, places=7)
