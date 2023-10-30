import numpy as np

from utils import ActivationFn, LossFn


class Layer:
	"""Represents one layer of a deep NN.  Weights and biases are stored in multidimensional numpy arrays"""

	def __init__(self, dims: tuple[int, int], activation: ActivationFn, neurotransmitters=1):
		self.dims = dims
		self.activation_fn = activation
		self.neurotransmitters = neurotransmitters
		self.weights = 2*np.random.rand(dims[1], dims[0], neurotransmitters)-1
		self.biases = 2*np.random.rand(dims[1], 1)-1
		self.last_inputs = None
		self.last_output = None

	def forward(self, inputs: np.ndarray):
		"""Performs a forward pass inference on this layer

		Args:
			inputs:
				The activations from the previous layer
		Returns:
			The activation from this layer"""

		self.last_inputs = inputs
		# Sim activations and add biases
		signal = np.dot(np.sum(self.weights, axis=2), inputs) + self.biases
		self.last_output = signal
		# Compute final activation using activation function
		return self.activation_fn.forward(signal)

	def backward(self, input_dx: np.ndarray, learning_rate: float):
		"""Performs a backward pass on this layer to update the paraeters

		Args:
			input_dx: The error partial derivative from the previous layer
			learning_rate: The learning rate of this layer
		Returns:
			The error for this layer to be passed further back"""

		shape = self.last_inputs.shape[1]

		output_dx = self.activation_fn.backward(input_dx, self.last_output)
		# Collapse neurotransmitters into the proper dimensions
		self.weights -= np.atleast_3d(learning_rate * np.dot(output_dx, self.last_inputs.T) / shape)
		self.biases -= learning_rate * np.sum(output_dx, axis=1, keepdims=True) / shape
		next_input_dx = np.dot(np.sum(self.weights, axis=2).T, output_dx)

		return next_input_dx

	def __repr__(self):
		return f"Layer({self.dims[0]}, {self.dims[1]})"


class Network:
	"""Class representing a set of neural networks, containing the weights and biases of one or more individual networks"""

	def __init__(self, layers: list[Layer], loss: LossFn):
		self.layers = layers
		self.loss = loss

	def forward(self, inputs: np.ndarray):
		"""Performs a forward inference pass for the entire network

		Args:
			inputs: The inputs to the network
		Returns:
			The output of the network"""

		layer_out = inputs
		for layer in self.layers:
			layer_out = layer.forward(layer_out)

		return layer_out

	def backward(self, output: np.ndarray, target: np.ndarray, learning_rate: float):
		"""Performs a complete backward pass of the network

		Args:
			output: The output of the network
			target: The target class of the sample
			learning_rate: The learning rate"""

		input_dx = np.gradient([output, target.T], axis=0)[0]

		for layer in reversed(self.layers):
			input_dx = layer.backward(input_dx, learning_rate)

	def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float = 1e-9):
		"""Trains this network on the given training samples

		Args:
			inputs: The inputs to train on
			targets: The target classes for the samples
			epochs: The number of epochs to train for
			learning_rate: The learning rate
		Returns:
			A list of the loss values for each epoch"""

		losses = []

		for i in range(epochs):
			output = self.forward(inputs.T)

			loss = self.loss(output, targets.T)
			losses.append(loss)

			self.backward(output, targets, learning_rate)

		return losses

	def __repr__(self):
		return f"Network({self.layers[0].dims[0]}, {', '.join([str(layer.dims[1]) for layer in self.layers])})"
