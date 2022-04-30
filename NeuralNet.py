import numpy as np
import json

import Utils


class NeuralNet:
	"""Class representing a neural network"""
	def __init__(self, topography, trainer):
		self.population = {"pop": []}
		self.topography = topography
		self.trainer = trainer
		# Initalize trainer
		self.trainer.prime(self.population, self.topography, self.loss)

	def train(self, sampleList, classList, epochs=1000, displayUpdate=10, verbosity=0, showPlots=False):
		self.trainer.train(sampleList, classList, epochs, displayUpdate, verbosity, showPlots)

	def test(self, sample, memberId):
		"""Returns the output of this neural network"""
		output = sample
		try:
			# Calculate output for each layer
			for layer in range(0, len(self.population["pop"][memberId])):
				output = Utils.sigmoid(np.dot(output, self.population["pop"][memberId][layer]))
		except Exception as e:
			print("ERROE: "+str(e))
		# Return final output of network
		return output

	def loss(self, samples, memberId=-1, verbosity=0):
		"""Calculates the loss of the network over the given sample set"""
		if memberId == -1:
			self.trainer.setAllFitness(samples)
			memberId = self.trainer.selection(1)[0]
		# Get output of network
		output = self.test(samples[0], memberId)
		outCopy = output
		correct = -1
		# Verbose logging
		if verbosity > 0:
			# Number of correct guesses
			correct = 0
			for i in range(0, len(output)):
				# Guesses class
				outChoice = np.where(outCopy[i] == max(outCopy[i]))[0][0]
				# Correct class
				correctChoice = np.where(samples[1][i] == max(samples[1][i]))[0][0]
				correctGuess = 1 if outChoice == correctChoice else 0
				if verbosity > 1:
					# Print out 5 most highly rated choices with their confidence
					for j in range(0, 5 if outCopy[i].shape[0] > 5 else outCopy[i].shape[0]):
						choiceCopy = np.where(outCopy[i] == np.amax(outCopy[i]))[0][0]
						print("Output choice: "+str(choiceCopy)+", Confidence: "+str(outCopy[i][choiceCopy]*100)+"%")
						# Eliminate choice to move onto next one
						outCopy[i][choiceCopy] = 0
					# Print colored text indicating a correct choice or not
					print(("\033[32m[=======CORRECT=======]\033[38m" if correctGuess == 1
							else "\033[93m[=======INCORRECT=======]\033[38m") + ", Correct choice: "+str(correctChoice))
				correct += correctGuess
		# Calculate sum of squared errors for loss function
		loss = 0.5 * np.sum((output - samples[1]) ** 2) / samples[1].shape[0]
		return loss, correct / len(output)

	def saveWeights(self, fileName):
		"""Saves state of the network to a file"""
		flatWeights = []
		for memberId in range(0, len(self.population["pop"])):
			for layer in range(0, len(self.population["pop"][memberId])):
				for weight in range(0, len(self.population["pop"][memberId][layer])):
					flatWeights.append(list(self.population["pop"][memberId][layer][weight]))

		with open(fileName, "w") as file:
			file.write(str(flatWeights))

	def loadWeights(self, fileName):
		"""Restores state of network from file"""
		with open(fileName, "r") as file:
			flatWeights = json.loads(file.read())
		for memberId in range(0, len(self.population["pop"])):
			for layer in range(0, len(self.population["pop"][memberId])):
				for weight in range(0, len(self.population["pop"][memberId][layer])):
					self.population["pop"][memberId][layer][weight] = np.array(flatWeights.pop(0))

	def __repr__(self):
		return "NeuralNetwork ("+str(self.topography)+")"
