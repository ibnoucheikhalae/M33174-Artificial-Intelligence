"""
Practical worksheet 5: a neural network trained using backpropagation.

Scope
-----
Use sigmoid activations in every trainable layer and half-sum squared-error
loss for one training example. Training will use one parameter update per
example, as in the worksheet. Biases are included here as an addition to the
worksheet, whose implementation contains weights but no biases.

Array and indexing conventions
------------------------------
All numerical arrays use NumPy float64. Single-example vectors are 1D arrays,
not 2D row or column arrays. Dataset arrays are 2D.

Let K = len(layer_sizes) - 1 be the number of trainable layers. Python layer
indices run from 0 to K, where layer 0 is the input layer. Connection indices
run from 0 to K - 1. Connection i joins source layer i to destination layer i+1.

For each connection index i:
    weights[i]          shape (layer_sizes[i], layer_sizes[i+1])
    biases[i]           shape (layer_sizes[i+1],)
    deltas[i]           shape (layer_sizes[i+1],)
    weight_gradients[i] shape identical to weights[i]
    bias_gradients[i]   shape identical to biases[i]

For each layer index k:
    activations[k]      shape (layer_sizes[k],)

Thus weights[i][s, d] connects source neuron s to destination neuron d.
There are K weight matrices and K bias vectors, but K+1 activation vectors.
The input layer has no biases or deltas of its own.

1. Forward pass
---------------
For i = 0, ..., K-1:
    z[i+1] = activations[i] @ weights[i] + biases[i]
    activations[i+1] = sigmoid(z[i+1])

Here @ means matrix multiplication. Multiplication by a scalar or between
same-shaped vectors is written with *. NumPy vector * vector is elementwise.

The sigmoid is:
    sigmoid(z) = 1 / (1 + exp(-z))

Using an already computed activation a = sigmoid(z):
    sigmoid_prime(z) = a * (1 - a)

Cache activations for the backward pass. The derivative helper accepts a,
not z, so storing the weighted inputs z is optional for this exercise.

2. Loss
-------
For one example with target vector t and prediction activations[K]:
    E = 0.5 * sum((activations[K] - t)**2)

This is a half-SUM, not a mean over the output neurons. If you change the
loss definition, its gradients must change consistently.

3. Backward pass: compute gradients
----------------------------------
Define deltas[i] = dE/dz[i+1]. A delta is a per-neuron loss gradient, not
merely the derivative of sigmoid and not a change in a weight.

Output layer:
    deltas[K-1] = (activations[K] - t) * activations[K] * (1 - activations[K])

Hidden layers, for i = K-2, ..., 0:
    propagated = deltas[i+1] @ weights[i+1].T
    deltas[i] = propagated * activations[i+1] * (1 - activations[i+1])

Parameter gradients, for i = 0, ..., K-1:
    weight_gradients[i] = outer(activations[i], deltas[i])
    bias_gradients[i] = deltas[i]

Here .T means transpose and outer means outer product. The source activation
comes first in the outer product to match (source, destination) weights.
Compute every gradient before changing any parameters.

4. Gradient-descent update
-------------------------
For each connection index i:
    weights[i] = weights[i] - learning_rate * weight_gradients[i]
    biases[i] = biases[i] - learning_rate * bias_gradients[i]

A learning rate is positive; it is not mathematically restricted to below 1.
Training repeats: forward pass -> loss -> backward pass -> parameter update.
An epoch is one complete pass through the training dataset.

Relationship to the worksheet's D and Der
-----------------------------------------
The worksheet's local D is a per-neuron delta. Its stored Der[i] has the
shape of a weight matrix and is formed from source activations and D.
Despite its comment, Der[i] is not just an activation derivative.

The worksheet starts with target - prediction, so its D and Der have the
opposite signs to the deltas and weight_gradients defined here. For the loss
above, its Der stores negative weight gradients, which explains why its GD
method ADDS the update. This file instead defines ordinary loss gradients
and SUBTRACTS them. Both sign conventions are valid when used consistently.

The worksheet reports mean squared error separately. For m output neurons,
that reported loss equals (2/m) * E; it is not generally the loss E above.
"""

import numpy as np

from typing import Tuple, Optional, List
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Because we need to store a lot of different types of data and associate between them, use a class
class NeuralNetwork:
    """
    A fully connected neural network with sigmoid activations and trainable biases.

    Constructor-created arrays contain initial parameters and zero-filled caches.
    Those caches do not represent an actual forward/backward pass until they have been filled in

    Attributes:
        layer_sizes: Widths of all layers, including the input and output
        weights: Matrices indexted by connetion, shaped (source, destination)
        biases: Bias vectors for the destination of each connection
        activations: Cached vectors for every layer, including the input
        deltas: Cached dE/dz vectors for trainable layers, indexed by connection
        weight_gradients: Cached dE/dW matrices, indexed by connection
        bias_gradients: Cached dE/db vectors, indexed by connection

    Example:
        network = NeuralNetwork(3, (4,), 2, seed=42)
        # Weight shapes: (3, 4), (4, 2)
        # Activation shapes: (3,), (4,), (2,)
    """

    def __init__(
            self,
            input_size: int = 2,
            hidden_sizes: Tuple[int, ...] = (2,2),
            output_size: int = 2,
            seed: Optional[int] = None
        ) -> None:
        """
        Intialise parameters and caches. And validate the architecture.

        Args:
            input_size: Positive number of input features per example
            hidden_sizes: Immutable tuple of hidden layer widths
            output_size: Positive number of output neurons
            seed: Optional nonnegative random seed for reproducable weights

        Weights are sampled uniformly from [0, 1), following the worksheet.
        Biases, activations, deltas, and parameter gradients are initialised to 0.
        All allocated arrays use float64 and belong to this network instance.

        Raises:
            TypeError: A layer width is not a Python int, or is a bool
            ValueError: A layer width is not positive, or the seed is invalid
        """
        # Total number of neurons in each layer. This creates a representation of the network in the
        # format we need it.
        # I.e., [n_neurons_inputs, n_neurons_hl_1, ..., n_neurons_hl_k, n_neurons_outputs]
        # default = [2, 2, 2, 2]
        layer_sizes = (input_size, *hidden_sizes, output_size)
        for size in layer_sizes:
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError("Every layer size must be an integer.")
            if size <= 0:
                raise ValueError("Every layer size must be positive.")

        self.layer_sizes = layer_sizes

        rng = np.random.default_rng(seed)

        # Initialise a weights array with random weights between 1 and 0
        # Weights array is indexed by connection. So weights[0] means the weights between the
        # input and HL1.
        # Each weight takes the shape (n_neurons_current_layer, n_neurons_next_layer), ensuring full
        # connectivity.
        self.weights: List[FloatArray] = [
            rng.random((source_size, destination_size), dtype=np.float64)
            for source_size, destination_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        # Intialise an empty cache to store biases. Indexed by the destination of each connection.
        # Therefore, biases[0] points to HL1.
        self.biases: List[FloatArray] = [
            np.zeros(size, dtype=np.float64) for size in layer_sizes[1:]
        ]

        # Initialise a cache to store activation values. Indexed by layer, including input layer.
        # Reminder: an activation is the neurons output, after applying the activation function
        # For one neuron: z = sum(wx) + b; a = sigmoid(z);
        # z = the weighted sum plus bias (pre-activation input);
        # a = the neurons output
        self.activations: List[FloatArray] = [
            np.zeros(size, dtype=np.float64) for size in layer_sizes
        ]

        # Initialise a cache to store the delta values. Indexed by connection.
        # Reminder: delta = dE/dz
        # This means ~"How sensitive is the loss `E` to a small change in this neuron's weighted
        # input `z`?"
        # A positive delta means increase `z` slightly increated the loss
        # A negative delta means increase `z` slightly decreases the loss
        self.deltas: List[FloatArray] = [
            np.zeros(size, dtype=np.float64) for size in layer_sizes[1:]
        ]

        # Initialise weight gradients cache. Indexed by weight.
        # Reminder: a weight gradient tells you how sensitive the loss is to that weight.
        # g = dE/dw
        # This is used in gradient descent: w_new = w_old + delta_w; delta_w = -lr * g
        self.weight_gradients: List[FloatArray] = [
            np.zeros_like(weights) for weights in self.weights
        ]

        # Initialse bias gradients cache. Indexed by bias.
        # This is the same as a weight gradient, but for biases.
        self.bias_gradients: List[FloatArray] = [
            np.zeros_like(biases) for biases in self.biases
        ]

    def feed_forward(self, inputs: FloatArray) -> FloatArray:
        """
        Compute a prediction for one example and cache every layer activation. - inference

        Args:
            inputs: Input data of shape (input_size,)

        Returns:
            Prediction vector of shape (output_size,)
            Only the final prediction is returned; intermediate activations put in self.activations

        Cache the input at activations[0] and each subsequent layer's output at activations[i+1].
        A new call replaces the activations needed by back_propagation for the given input.
        """
        activation = inputs.copy()
        self.activations[0] = activation

        for i, weights in enumerate(self.weights):
            weighted_inputs = activation @ weights + self.biases[i]  # z
            activation = self.sigmoid(weighted_inputs)  # a

            self.activations[i + 1] = activation

        # Final activation is output layer
        return activation

    def back_propagation(self, targets: FloatArray) -> None:
        """
        Compute and store parameter gradients for the latest forward pass.

        Args:
            targets: Target vector of shape (output_size,) for the input supplied to the last
            `feed_forward` call

        Requires a preceeding forward pass with the current weights and biases.
        Uses mean squared error averaged across the output neurons.

        Replace deltas, weight_gradients, and bias_gradients without updating any parameters.
        The `step` method applies the updates afterwards (see `train_nn`).
        """
        # Prediction is the last activation -- output layer
        predictions = self.activations[-1]

        # Derivative of MSE with respect to each output neuron:
        #   dE/da = 2/m * (a - t)
        # This means ~"How sensitive the loss is to each output activation: dE/da"
        activation_gradient = (2.0 / targets.size) * (predictions - targets)

        # Go through the weights in reverse
        for i in reversed(range(len(self.weights))):
            # Connection i feeds desination layer i+1
            activation = self.activations[i + 1]

            # dE/dz =        dE/da        *         da/dz
            # delta = activation_graident * activation_derivative
            activation_derivative = self.sigmoid_derivative_from_activation(activation)
            delta = activation_gradient * activation_derivative
            # For intuition of the above, if sigmoid is nearly flat at this neuron, chaning `z`
            # barely changes it activation. That small sigmoid derivative reduces how strongly
            # a change in `z` affects the loss.
            # Delta does not tell us how much to update a weight. For a particular connection:
            #     weight_gradient   = source_activation * destination_delta
            #     weight_adjustment = -learning_rate * weight_gradient  (performed in `step`)

            self.deltas[i] = delta

            # np.outer multiplies every source activation by every destination delta
            # Example:
            #     source_activations = [2, 3];  destination_deltas = [0.1, -0.2]
            #     outer_product = [
            #         [2 * 0.1, 2 * -0.2],
            #         [3 * 0.1, 3 * -0.2]
            #     ]
            # This gives us our expected shape, where each entry is the gradient for one
            # source-to-destination weight
            self.weight_gradients[i] = np.outer(self.activations[i], delta)

            # Bias gradients have the same numerical values as deltas because a neuron adds its
            # bias directly: z = aw + b
            # Therefore, increasing `b` by a small amount increases `z` by exactly that amount
            # So although self.deltas and self.bias_gradients have different semantic meaning,
            # their values coincide. And therefore we can avoid doing repeated arithmetic here.
            self.bias_gradients[i] = delta.copy()

            if i > 0:  # Don't update on input layer
                # Propagate the loss gradient to the preceding trainable layer
                # .T transposes -- swapping rows and columns
                activation_gradient = delta @ self.weights[i].T

    def step(self, learning_rate: float) -> None:
        """
        Apply one gradient-descent update using the stored parameter gradients.

        Args:
            learning_rate: Positive, finite step size for this update

        Requires gradients from back propagation evaluated at the current parameters.
        Update both weights and biases, subtracting their scaled loss gradients.
        Applying an update makes the preceding forward cache and gradients stale; calculate a new
        forward/backward pass before the next step.
        """

        # Reminder: a weight gradient tells you how sensitive the loss is to that weight.
        # g = dE/dw
        # This is used in gradient descent: w_new = w_old + delta_w; delta_w = -lr * g
        for i, old_biases in enumerate(self.biases):
            delta_bias = -learning_rate * self.bias_gradients[i]
            new_biases = old_biases + delta_bias
            self.biases[i] = new_biases

        for i, old_weights in enumerate(self.weights):
            delta_weights = -learning_rate * self.weight_gradients[i]
            new_weights = old_weights + delta_weights
            self.weights[i] = new_weights


    def train_nn(
            self,
            inputs: FloatArray,
            targets: FloatArray,
            epochs: int,
            learning_rate: float
        ) -> List[float]:
        """
        Method to train the network on a dataset, updating parameters after each example

        Args:
            inputs: Input matrix of shape (number_of_examples, input_size)
            targets: Target matrix of shape (number_of_examples, output_size)
            epochs: Nonnegative number of complete passes through the dataset
            learning_rate: Positive, finite step size for each parameter update

        Returns:
            A list of the average loss per epoch, showing progression over time

        Corresponding rows of inputs and targets describe the same example.
        Process examples in their supplied order.
        Each epoch trains on every example and stores the average loss across all of them.
        This method coordinates forward, loss, backward, and step without duplicating their
        responsibilities.
        """
        epoch_mean_losses: List[float] = []
        for epoch in range(epochs):

            epoch_loss_sum = 0

            for i, example_input in enumerate(inputs):
                example_target = targets[i]

                # Do the thing
                predictions = self.feed_forward(example_input)
                self.back_propagation(example_target)
                self.step(learning_rate)

                epoch_loss_sum += self.loss(example_target, predictions)

            epoch_mean_losses.append(epoch_loss_sum / inputs.shape[0])

        return epoch_mean_losses

    @staticmethod
    def sigmoid(z: FloatArray) -> FloatArray:
        """
        Apply sigmoid elementwie, prserving the input array's shape.

        Args:
            z: Weighted input values before activation

        Returns:
            Sigmoid activations with the same shape as z
        """
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative_from_activation(activation: FloatArray) -> FloatArray:
        """
        Evaluate the sigmoid derivative using an already computed activation

        Args:
            activation: Sigmoid output a, not the pre-activation weighted input z

        Returns:
            Derivative values of the activation, with the same shape as the activation

        The result is sigmoid'(z), evaluated from
        a = sigmoid(z); sigmoid_derivative_from_activation(a);
        """
        return activation * (1.0 - activation)

    @staticmethod
    def loss(targets: FloatArray, predictions: FloatArray) -> float:
        """
        Calculate the mean square error for one example.

        Args:
            targets: Target vector of shape output_size
            predictions: Prediction vector with exact same shape

        Returns:
            A float generated from the mean sqare error loss function
            This means average across the output neurons
        """
        return np.average((targets - predictions) ** 2)

def main() -> int:
    breakpoint()
    my_nn = NeuralNetwork()
    loss_by_epochs = my_nn.train_nn()

    return 0

if __name__ == "__main__":
    main()