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
    load_phase_network,
    train_phase_network,
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
        network = NeuralNetwork(5, (), 24, seed=1)
        network.weights[0][:] = 0
        network.biases[0][:] = 0
        network.weights[0][1, 0] = 2
        inputs = create_phase_inputs(300)
        predictions = np.array([network.feed_forward(x) for x in inputs])
        expected = MIN_JOINT_ANGLES + predictions.reshape(300, 8, 3) * (
            MAX_JOINT_ANGLES - MIN_JOINT_ANGLES
        )
        np.testing.assert_array_equal(generate_phase_gait(network), expected)
        # Each frame uses its own phase, including frame zero; there is no seed pose.
        self.assertGreater(np.ptp(expected[:, 0, 0]), 0.1)
        self.assertTrue((expected >= MIN_JOINT_ANGLES).all())
        self.assertTrue((expected <= MAX_JOINT_ANGLES).all())
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
            gait_path, model_path = folder / "gait.npy", folder / "model.pkl"
            for stored in (gait, gait[None]):
                np.save(gait_path, stored)
                with (
                    patch.object(NeuralNetwork, "train_nn", record_training),
                    redirect_stdout(io.StringIO()),
                ):
                    network = train_phase_network(gait_path, model_path, epochs=1)
                    loaded = load_phase_network(model_path)
                self.assertEqual(network.layer_sizes, (5, 64, 24))
                np.testing.assert_array_equal(
                    generate_phase_gait(network), generate_phase_gait(loaded)
                )
                self.assertTrue(
                    (folder / "training_loss.png").read_bytes().startswith(b"\x89PNG")
                )
            expected = (
                (gait - MIN_JOINT_ANGLES) / (MAX_JOINT_ANGLES - MIN_JOINT_ANGLES)
            ).reshape(300, 24)
            for targets in seen_targets:
                np.testing.assert_array_equal(targets, expected)
            for invalid in (
                gait[:299],
                np.full_like(gait, np.nan),
                np.full_like(gait, 10),
            ):
                np.save(gait_path, invalid)
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    train_phase_network(gait_path, model_path, epochs=1)

    def test_load_phase_network(self):
        """Preserve the original compatibility checks for saved models."""
        network = NeuralNetwork(5, (), 24, seed=1)
        saved = {
            "format": "phase_pose_v1",
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
                {"weights": []},
                {"biases": [np.full(24, np.nan)]},
            ):
                with path.open("wb") as file:
                    pickle.dump({**saved, **change}, file)
                with self.subTest(change=change.keys()), self.assertRaises(ValueError):
                    load_phase_network(path)
