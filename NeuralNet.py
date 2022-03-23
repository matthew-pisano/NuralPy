# import the necessary packages
import numpy as np
import json

from matplotlib import pyplot as plt

import Utils


class NeuralNet:
	def __init__(self, topography, trainer):
		self.population = []
		self.topography = topography
		self.trainer = trainer
		self.trainer.prime(self.population, self.topography, self.loss)

	def train(self, sampleList, classList, epochs=1000, learningRate=0.1, displayUpdate=10, verbosity=0):
		self.trainer.train(sampleList, classList, epochs, learningRate, displayUpdate, verbosity)

	def test(self, sample, memberId):
		# Change into 2D array
		output = np.atleast_2d(sample)
		try:
			for layer in range(0, len(self.population[memberId])):
				output = Utils.sigmoid(np.dot(output, self.population[memberId][layer]))
		except Exception as e:
			print("ERROE: "+str(e))
		return output

	def loss(self, samples, memberId=0, verbosity=0):
		# Change into 2D array
		samples[1] = np.atleast_2d(samples[1])
		output = self.test(samples[0], memberId)
		outCopy = output
		correct = 0
		if verbosity > 0:
			for i in range(0, len(output)):
				outChoice = np.where(outCopy[i] == max(outCopy[i]))[0][0]
				correctChoice = np.where(samples[1][i] == max(samples[1][i]))[0][0]
				if verbosity > 1:
					for j in range(0, 5):
						choiceCopy = np.where(outCopy[i] == np.amax(outCopy[i]))[0][0]
						print("Output choice: "+str(choiceCopy)+", Confidence: "+str(outCopy[i][choiceCopy]*100)+"%")
						outCopy[i][choiceCopy] = 0
					print("Correct choice: "+str(correctChoice))
				correct += 1 if outChoice == correctChoice else 0
		# Calculate sum of squared errors for loss function
		return 0.5 * np.sum((output - samples[1]) ** 2), correct / len(output)

	def saveWeights(self, fileName):
		flatWeights = []
		for memberId in range(0, len(self.population)):
			for layer in range(0, len(self.population[memberId])):
				for weight in range(0, len(self.population[memberId][layer])):
					flatWeights.append(list(self.population[memberId][layer][weight]))

		with open(fileName, "w") as file:
			file.write(str(flatWeights))

	def loadWeights(self, fileName):
		with open(fileName, "r") as file:
			flatWeights = json.loads(file.read())
		for memberId in range(0, len(self.population)):
			for layer in range(0, len(self.population[memberId])):
				for weight in range(0, len(self.population[memberId][layer])):
					self.population[memberId][layer][weight] = np.array(flatWeights.pop(0))

	def __repr__(self):
		return "NeuralNetwork ("+str(self.topography)+")"
