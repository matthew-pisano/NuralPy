# import the necessary packages
import numpy as np

import Utils


class NeuralNetwork:
	def __init__(self, topography, learningRate=0.1):
		self.weights = []
		self.topography = topography
		self.learningRate = learningRate
		# Initialize weights and thresholds for all but last layer
		for i in range(0, len(topography) - 2):
			weight = np.random.randn(topography[i] + 1, topography[i + 1] + 1)
			self.weights.append(weight / np.sqrt(topography[i]))
		# Initialize weights and thresholds for output layer
		weight = np.random.randn(topography[-2] + 1, topography[-1])
		self.weights.append(weight / np.sqrt(topography[-2]))

	def train(self, sampleList, classList, epochs=1000, displayUpdate=10):
		# Append column to 1's to allow for training thresholds
		sampleList = np.c_[sampleList, np.ones((sampleList.shape[0]))]
		for epoch in range(0, epochs):
			# Train on each sample
			for (sample, classOf) in zip(sampleList, classList):
				self.epochTrain(sample, classOf)
			# check to see if we should display a training update
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.loss(sampleList, classList)
				print("Epoch: "+str(epoch)+", Loss: "+str(loss))

	def epochTrain(self, sample, classOf):
		# Change into 2D array
		activations = [np.atleast_2d(sample)]
		# Gather activations for each layer
		for layer in range(0, len(self.weights)):
			activation = activations[layer].dot(self.weights[layer])
			activations.append(Utils.sigmoid(activation))
		# Calculate error of output layer
		error = activations[-1] - classOf
		deltas = [error * Utils.sigmoidDx(activations[-1])]

		for layer in range(len(activations) - 2, 0, -1):
			delta = deltas[-1].dot(self.weights[layer].transpose())
			delta = delta * Utils.sigmoidDx(activations[layer])
			deltas.append(delta)
		# Update weights
		for layer in range(0, len(self.weights)):
			self.weights[layer] += -self.learningRate * activations[layer].transpose().dot(deltas[-(layer+1)])

	def predict(self, sample, useThreshold=True):
		# Change into 2D array
		output = np.atleast_2d(sample)
		if useThreshold:
			# Append column to 1's to allow for training thresholds
			output = np.c_[output, np.ones((output.shape[0]))]

		for layer in range(0, len(self.weights)):
			output = Utils.sigmoid(np.dot(output, self.weights[layer]))
		return output

	def loss(self, sampleList, classList):
		# Change into 2D array
		classList = np.atleast_2d(classList)
		output = self.predict(sampleList, useThreshold=False)
		# Calculate sum of squared errors for loss function
		return 0.5 * np.sum((output - classList) ** 2)

	def __repr__(self):
		return "NeuralNetwork ("+str(self.topography)+")"
