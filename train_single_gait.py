
from pathlib import Path
import numpy as np
from NN.nn_template import save_model

from NN.nn_template import (
    validate_gaits,
    build_training_dataset,
    build_network,
    train_network,
    generate_gait,
    save_model,
)

# Load exploratory GA data
with np.load(
    "NN/data_exploratory/run_best_gait.npz",
    allow_pickle=False
) as data:
    gaits = validate_gaits(data["gaits"])

# Convert gait to input/target transitions
inputs, targets = build_training_dataset(gaits)

print("Inputs:", inputs.shape)
print("Targets:", targets.shape)

# Build and train the neural network
network = build_network(seed=42)

losses = train_network(
    network,
    inputs,
    targets,
    epochs=3,
    learning_rate=0.1,
    on_epoch=lambda epoch, mse: print(
        f"Epoch {epoch}: MSE={mse:.8f}"
    ),
)

# Save trained model and generated gait
output = Path("nn_results/single_gait_test")
output.mkdir(parents=True, exist_ok=True)
# Save the trained neural network
save_model(
    network,
    output / "phase_model.pkl"
)

print("Trained NN model saved successfully!")

save_model(network, output / "model.pkl")

generated = generate_gait(network, gaits[0, 0])
np.save(output / "generated_gaits.npy", generated[None])

print("Training complete!")
print("Final training MSE:", losses[-1])
