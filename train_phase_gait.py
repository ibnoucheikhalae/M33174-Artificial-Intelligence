
from pathlib import Path
import numpy as np

from NN.NN import NeuralNetwork
from NN.nn_template import (
    validate_gaits,
    normalise_joint_angles,
    denormalise_joint_angles,
)
from GA.GA import evaluate_fitness

# Load the verified GA gait
ga = validate_gaits(
    np.load("ga_training_check.npy")
)[0]

# Each frame gets a time/phase representation
t = np.linspace(0, 1, len(ga))

inputs = np.column_stack([
    t,
    np.sin(2 * np.pi * t),
    np.cos(2 * np.pi * t),
    np.sin(4 * np.pi * t),
    np.cos(4 * np.pi * t),
])

# Targets are the actual GA joint positions
targets = normalise_joint_angles(ga).reshape(300, 24)

# Use the team's existing NeuralNetwork class
network = NeuralNetwork(
    input_size=5,
    hidden_sizes=(64,),
    output_size=24,
    seed=42,
)

# Xavier initialisation
rng = np.random.default_rng(42)
for i, weights in enumerate(network.weights):
    source, destination = weights.shape
    limit = np.sqrt(6 / (source + destination))
    network.weights[i] = rng.uniform(
        -limit, limit, weights.shape
    )

# Train
def report(epoch, mse):
    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch}: MSE={mse:.8f}")

network.train_nn(
    inputs,
    targets,
    epochs=300,
    learning_rate=0.1,
    on_epoch=report,
)

# Generate the gait directly from each frame's phase
predictions = np.array([
    network.feed_forward(x) for x in inputs
])

generated = denormalise_joint_angles(
    predictions
).reshape(300, 8, 3)

output = Path("nn_results/phase_experiment")
output.mkdir(parents=True, exist_ok=True)

from NN.nn_template import save_model

save_model(network, output / "phase_model.pkl")

print("Trained NN model saved successfully!")

np.save(
    output / "generated_gaits.npy",
    generated[None]
)

print("\n--- FINAL RESULTS ---")
print("GA fitness:", evaluate_fitness(ga[None])[0])
print(
    "NN fitness:",
    evaluate_fitness(generated[None])[0]
)
print(
    "Pose MSE:",
    np.mean((predictions - targets) ** 2)
)
print(
    "NN last 50 frames movement:",
    np.ptp(generated[-50:], axis=0).mean()
)
print(
    "GA last 50 frames movement:",
    np.ptp(ga[-50:], axis=0).mean()
)
