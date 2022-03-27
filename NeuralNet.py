# import the necessary packages
import time

import numpy as np
import json

from matplotlib import pyplot as plt

import Utils


class NeuralNet:
	def __init__(self, topography, trainer):
		self.population = {"pop": []}
		self.topography = topography
		self.trainer = trainer
		self.trainer.prime(self.population, self.topography, self.loss)

	def train(self, sampleList, classList, epochs=1000, displayUpdate=10, verbosity=0):
		self.trainer.train(sampleList, classList, epochs, displayUpdate, verbosity)

	def test(self, sample, memberId):
		# Change into 2D array
		# output = np.atleast_2d(sample)
		output = sample
		try:
			for layer in range(0, len(self.population["pop"][memberId])):
				output = Utils.sigmoid(np.dot(output, self.population["pop"][memberId][layer]))
		except Exception as e:
			print("ERROE: "+str(e))
		return output

	def loss(self, samples, memberId=-1, verbosity=0):
		if memberId == -1:
			self.trainer.setAllFitness(samples)
			memberId = self.trainer.selection(1)[0]
		# timeAt = time.time()
		# Change into 2D array
		# samples[1] = np.atleast_2d(samples[1])
		output = self.test(samples[0], memberId)
		outCopy = output
		correct = -1
		if verbosity > 0:
			correct = 0
			for i in range(0, len(output)):
				outChoice = np.where(outCopy[i] == max(outCopy[i]))[0][0]
				correctChoice = np.where(samples[1][i] == max(samples[1][i]))[0][0]
				correctGuess = 1 if outChoice == correctChoice else 0
				if verbosity > 1:
					for j in range(0, 5 if outCopy[i].shape[0] > 5 else outCopy[i].shape[0]):
						choiceCopy = np.where(outCopy[i] == np.amax(outCopy[i]))[0][0]
						print("Output choice: "+str(choiceCopy)+", Confidence: "+str(outCopy[i][choiceCopy]*100)+"%")
						outCopy[i][choiceCopy] = 0
					print(("\033[32m[=======CORRECT=======]\033[38m" if correctGuess == 1
						else "\033[93m[=======INCORRECT=======]\033[38m") + ", Correct choice: "+str(correctChoice))
				correct += correctGuess
		# print("Timeat: "+str(time.time()-timeAt))
		# timeAt = time.time()
		# Calculate sum of squared errors for loss function
		loss = 0.5 * np.sum((output - samples[1]) ** 2) / samples[1].shape[0]
		#print("Timeat: " + str(time.time() - timeAt))
		return loss, correct / len(output)

	def saveWeights(self, fileName):
		flatWeights = []
		for memberId in range(0, len(self.population["pop"])):
			for layer in range(0, len(self.population["pop"][memberId])):
				for weight in range(0, len(self.population["pop"][memberId][layer])):
					flatWeights.append(list(self.population["pop"][memberId][layer][weight]))

		with open(fileName, "w") as file:
			file.write(str(flatWeights))

	def loadWeights(self, fileName):
		with open(fileName, "r") as file:
			flatWeights = json.loads(file.read())
		for memberId in range(0, len(self.population["pop"])):
			for layer in range(0, len(self.population["pop"][memberId])):
				for weight in range(0, len(self.population["pop"][memberId][layer])):
					self.population["pop"][memberId][layer][weight] = np.array(flatWeights.pop(0))

	def __repr__(self):
		return "NeuralNetwork ("+str(self.topography)+")"
