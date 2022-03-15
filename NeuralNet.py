# import the necessary packages
import numpy as np
import json

from matplotlib import pyplot as plt

import Utils


class NeuralNet:
	def __init__(self, topography, trainer):
		self.weights = []
		self.topography = topography
		# Initialize weights and thresholds for all but last layer
		for i in range(0, len(topography) - 2):
			weight = np.random.randn(topography[i] + 1, topography[i + 1] + 1)
			self.weights.append(weight / np.sqrt(topography[i]))
		# Initialize weights and thresholds for output layer
		weight = np.random.randn(topography[-2] + 1, topography[-1])
		self.weights.append(weight / np.sqrt(topography[-2]))
		self.trainer = trainer
		self.trainer.prime(self.weights, self.loss)

	def train(self, sampleList, classList, epochs=1000, learningRate=0.1, displayUpdate=10):
		self.trainer.train(sampleList, classList, epochs, learningRate, displayUpdate)

	def test(self, sample):
		# Change into 2D array
		output = np.atleast_2d(sample)
		try:
			for layer in range(0, len(self.weights)):
				output = Utils.sigmoid(np.dot(output, self.weights[layer]))
		except Exception as e:
			pass
		return output

	def loss(self, sampleList, classList, display=False):
		# Change into 2D array
		classList = np.atleast_2d(classList)
		output = self.test(sampleList)
		outCopy = output
		correct = 0
		for i in range(0, len(output)):
			outChoice = np.where(outCopy[i] == max(outCopy[i]))[0][0]
			correctChoice = np.where(classList[i] == max(classList[i]))[0][0]
			if display:
				for j in range(0, 5):
					choiceCopy = np.where(outCopy[i] == np.amax(outCopy[i]))[0][0]
					print("Output choice: "+str(choiceCopy)+", Confidence: "+str(outCopy[i][choiceCopy]*100)+"%")
					outCopy[i][choiceCopy] = 0
				print("Correct choice: "+str(correctChoice))
				"""reconstruct = Utils.toImage(sampleList[i], 28)
				plt.imshow(np.array(reconstruct), cmap="gray")
				plt.show()"""
			correct += 1 if outChoice == correctChoice else 0
		# Calculate sum of squared errors for loss function
		return 0.5 * np.sum((output - classList) ** 2), correct / len(output)

	def saveWeights(self, fileName):
		flatWeights = []
		for layer in range(0, len(self.weights)):
			for weight in range(0, len(self.weights[layer])):
				flatWeights.append(list(self.weights[layer][weight]))

		with open(fileName, "w") as file:
			file.write(str(flatWeights))

	def loadWeights(self, fileName):
		with open(fileName, "r") as file:
			flatWeights = json.loads(file.read())
		for layer in range(0, len(self.weights)):
			for weight in range(0, len(self.weights[layer])):
				self.weights[layer][weight] = np.array(flatWeights.pop(0))

	def __repr__(self):
		return "NeuralNetwork ("+str(self.topography)+")"
