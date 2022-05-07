import random

import Utils


class Neuron:
    """Class representing one neuron of a neural network"""
    def __init__(self, id, prevLayer):
        """Initialize neuron"""
        self.threshold = 0
        self.id = id
        self.prevLayerNodes = prevLayer
        self.sum = 0
        self.signalsGotten = 0
        self.activationFunction = Utils.tanh
        self.connections = []
        self.weights = {}
        self.errorGradient = 0

    def getId(self):
        """Returns the id of the neuron"""
        return self.id

    def getSum(self):
        """Returns the sum of the weighted inputs to the network"""
        return self.sum

    def setConnections(self, outputs):
        """Connects this neuron to all others in the output list"""
        self.connections = outputs

    def getErrorGradient(self):
        """Returns the error gradient"""
        return self.errorGradient

    def getWeight(self, neuronId):
        """Returns the weight"""
        return self.weights[neuronId]

    def mapWeights(self, inputs):
        """Initializes all weights to zero"""
        for n in inputs:
            self.weights[n.id] = [0.0, 0.0]

    def initialize(self):
        """Set weights and threshold to random values"""
        self.threshold = random.uniform(-.5, .5)
        for weight in self.weights.values():
            weight[0] = random.uniform(-.5, .5)

    def sendSignal(self, signal, target):
        """SEnt the output to the target neuron"""
        target.getSignal(signal, self.id)

    def getSignal(self, signal, senderId):
        """Gets the signal from an input neuron and fires this neuron if this is
        the last signal needed"""
        if self.signalsGotten == 0:
            self.sum = 0
        if senderId != -1:
            self.sum += self.weights[senderId][0] * signal
            self.weights[senderId][1] = signal
        else:
            self.sum += signal
        self.signalsGotten += 1
        if self.signalsGotten == self.prevLayerNodes:
            self.signalsGotten = 0
            self.fire()

    def activationOutput(self):
        """Gets the sum of the weights multiplied by their latest output"""
        totalSignal = 0
        for weight in self.weights.values():
            totalSignal += weight[1] * weight[0]
        return self.activationFunction(totalSignal - self.threshold)

    def correctOutWeights(self, error, learningRate):
        """Corrects each weight based on the output error gradient"""
        output = self.activationOutput()
        self.errorGradient = output * (1-output) * error
        for weight in self.weights.values():
            # correct weight
            weight[0] += learningRate * self.errorGradient * weight[1]

    def correctWeights(self, learningRate):
        """Corrects the weights based on the hidden layer output gradient"""
        nextResults = 0
        for n in self.connections:
            nextResults += n.getErrorGradient() * n.getWeight(self.id)[0]
        output = self.activationOutput()
        self.errorGradient = output * (1-output) * nextResults
        for weight in self.weights.values():
            # correct weight
            weight[0] += learningRate * self.errorGradient * weight[1]

    def __repr__(self):
        return str(self.id)

    def fire(self):
        """Sends the activation function output to all the connections to this neuron"""
        signal = self.activationFunction(self.sum - self.threshold)
        for n in self.connections:
            self.sendSignal(signal, n)

