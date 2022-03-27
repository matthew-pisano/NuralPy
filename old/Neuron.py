import random

import Utils


class Neuron:

    def __init__(self, id, prevLayer):
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
        return self.id

    def getSum(self):
        return self.sum

    def setConnections(self, outputs):
        self.connections = outputs

    def getErrorGradient(self):
        return self.errorGradient

    def getWeight(self, neuronId):
        return self.weights[neuronId]

    def mapWeights(self, inputs):
        for n in inputs:
            self.weights[n.id] = [0.0, 0.0]

    def initialize(self):
        self.threshold = random.uniform(-.5, .5)
        for weight in self.weights.values():
            weight[0] = random.uniform(-.5, .5)

    def sendSignal(self, signal, target):
        target.getSignal(signal, self.id)

    def getSignal(self, signal, senderId):
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
        totalSignal = 0
        for weight in self.weights.values():
            totalSignal += weight[1] * weight[0]
        return self.activationFunction(totalSignal - self.threshold)

    def correctOutWeights(self, error, learningRate):
        output = self.activationOutput()
        self.errorGradient = output * (1-output) * error
        for weight in self.weights.values():
            # correct weight
            weight[0] += learningRate * self.errorGradient * weight[1]

    def correctWeights(self, learningRate):
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
        signal = self.sum - self.threshold
        signal = self.activationFunction(self.sum - self.threshold)
        for n in self.connections:
            self.sendSignal(signal, n)

