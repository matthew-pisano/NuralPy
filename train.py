import numpy as np

import Utils


class Trainer:
    def __init__(self):
        self.weights = None
        self.loss = None

    def prime(self, weights, loss):
        self.weights = weights
        self.loss = loss

    def train(self, sampleList, classList, epochs=1000, learningRate=0.1, displayUpdate=10):
        pass

    def epochTrain(self, sample, classOf, learningRate=0.1):
        pass


class Backpropogator(Trainer):
    def train(self, sampleList, classList, epochs=1000, learningRate=0.1, displayUpdate=10):
        # Append column to 1's to allow for training thresholds
        sampleList = np.c_[sampleList, np.ones((sampleList.shape[0]))]
        for epoch in range(0, epochs):
            # Train on each sample
            for (sample, classOf) in zip(sampleList, classList):
                self.epochTrain(sample, classOf, learningRate)
            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.loss(sampleList, classList)
                print("Epoch: " + str(epoch) + ", Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")

    def epochTrain(self, sample, classOf, learningRate=0.1):
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
            self.weights[layer] += -learningRate * activations[layer].transpose().dot(deltas[-(layer + 1)])


class Genetic(Trainer):
    def __init__(self):
        self.weights = None
        self.loss = None