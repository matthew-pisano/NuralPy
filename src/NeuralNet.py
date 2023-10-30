import numpy as np
import json

import Utils


class NeuralNet:
	"""Class representing a set of neural networks,
	containing the weights and biases of one or more individual networks"""
	def __init__(self, topography, trainer):
		# List of on or more neural networks with their weights and biases stored inside numpy arrays
		self.population = {"pop": []}
		self.topography = topography
		self.trainer = trainer
		# Initialize trainer
		self.trainer.prime(self.population, self.topography, self.loss)

	def train(self, sampleList, classList, epochs: int = 1000, displayUpdate: int = 10, verbosity: int = 0, showPlots: bool = False):
		"""Trains the neural network with the given parameters along with progress reports for logging"""
		# displayUpdate indicates after how many epochs a progress update will be displayed,
		# showPlots tells whether to show the loss and accuracy graphs
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

	def loss(self, samples, memberId=-1, verbosity=0, displaySamples=None):
		"""Calculates the loss of the network over the given sample set and will output useful logging information"""
		# memberId is the id of the individual to test, only useful for genetic algorithms,
		# displaySamples is the numpy array to show during testing.  It is useful to have it
		# be the original data set before any normalization for better viewing
		# samples[0] is a list of samples with their attributes, samples[1] is a list of the corresponding classes
		if memberId == -1:
			# For genetic algorithm, set fittnesses of individuals
			self.trainer.setAllFitness(samples)
			memberId = self.trainer.selection(1)[0]
		# Get output of network for all samples
		output = self.test(samples[0], memberId)
		correct = -1
		# Verbose logging
		if verbosity > 0:
			# Number of correct guesses
			correct = 0
			outCopy = np.copy(output)
			for i in range(0, len(output)):
				# Guesses class
				outChoice = np.where(outCopy[i] == max(outCopy[i]))[0][0]
				# Correct class
				correctChoice = np.where(samples[1][i] == max(samples[1][i]))[0][0]
				correctGuess = 1 if outChoice == correctChoice else 0
				if verbosity > 1:
					# For linux Green color: \033[32m, Yellow Color: \033[93m, Default: \033[38m
					# Print text indicating a correct choice or not
					print(("[=======CORRECT=======]" if correctGuess == 1 else "[xxxxxxxINCORRECTxxxxxxx]") +
						  ", Correct choice: " + str(correctChoice))
					# Print out 5 most highly rated choices with their confidence
					for j in range(0, 5 if outCopy[i].shape[0] > 5 else outCopy[i].shape[0]):
						choiceCopy = np.where(outCopy[i] == np.amax(outCopy[i]))[0][0]
						print("Output choice: "+str(choiceCopy)+", Confidence: "+str(outCopy[i][choiceCopy]*100)+"%")
						# Eliminate choice to move onto next one
						outCopy[i][choiceCopy] = 0
					if verbosity > 2:
						if not displaySamples:
							print("Sample Attributes: "+str(samples[0][i].tolist()[:-1]))
						else:
							print("Sample Attributes: " + str(displaySamples[0][i].tolist()))
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
