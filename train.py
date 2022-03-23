import random
import time

import numpy as np
import math
import Utils


class Trainer:
    def __init__(self):
        self.population = None
        self.loss = None
        self.topography = None

    def prime(self, population, topography, loss):
        self.population = population
        self.loss = loss
        self.topography = topography

    def train(self, sampleList, classList, epochs=1000, learningRate=0.1, displayUpdate=10, verbosity=0):
        # Append column to 1's to allow for training thresholds
        sampleList = np.c_[sampleList, np.ones((sampleList.shape[0]))]
        for epoch in range(0, epochs):
            # Train on each sample
            sampleNum = 0
            timeAt = time.time()
            for (sample, classOf) in zip(sampleList, classList):
                self.epochTrain(sample, classOf, learningRate)
                sampleNum += 1
                print("Sample "+str(sampleNum)+" completed after "+str(time.time()-timeAt)+"s")
                timeAt = time.time()
            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                totalLoss = [0, 0]
                bestLoss = [0, 0]
                mostCorrect = [0, 0]
                for i in range(0, len(self.population)):
                    loss = self.loss([sampleList, classList], verbosity=verbosity)
                    totalLoss[0] += loss[0]
                    totalLoss[1] += loss[1]
                    if loss[0] > bestLoss[0]:
                        bestLoss = loss
                    if loss[1] > mostCorrect[1]:
                        mostCorrect = loss
                totalLoss[0] /= len(self.population)
                print("Epoch: " + str(epoch) + ", Loss: " + str(totalLoss[0]) + (
                            ", Correct: " + str(totalLoss[1] * 100) + "%" if verbosity > 0 else ""), end="")
                if len(self.population) > 1:
                    print(", Best loss: "+str(bestLoss)+", most correct: "+str(mostCorrect))
                else:
                    print()

    def epochTrain(self, sample, classOf, learningRate=0.1):
        pass

    def initMember(self):
        self.population.append([])
        # Initialize weights and thresholds for all but last layer
        for i in range(0, len(self.topography) - 2):
            weight = np.random.randn(self.topography[i] + 1, self.topography[i + 1] + 1)
            self.population[-1].append(weight / np.sqrt(self.topography[i]))
        # Initialize weights and thresholds for output layer
        weight = np.random.randn(self.topography[-2] + 1, self.topography[-1])
        self.population[-1].append(weight / np.sqrt(self.topography[-2]))


class Backpropogator(Trainer):
    def prime(self, population, topography, loss):
        super().prime(population, topography, loss)
        super().initMember()

    def epochTrain(self, sample, classOf, learningRate=0.1):
        # Change into 2D array
        activations = [np.atleast_2d(sample)]
        # Gather activations for each layer
        for layer in range(0, len(self.population[0])):
            activation = activations[layer].dot(self.population[0][layer])
            activations.append(Utils.sigmoid(activation))
        # Calculate error of output layer
        error = activations[-1] - classOf
        deltas = [error * Utils.sigmoidDx(activations[-1])]

        for layer in range(len(activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.population[0][layer].transpose())
            delta = delta * Utils.sigmoidDx(activations[layer])
            deltas.append(delta)
        # Update weights
        for layer in range(0, len(self.population[0])):
            self.population[0][layer] += -learningRate * activations[layer].transpose().dot(
                deltas[-(layer + 1)])


class Genetic(Trainer):
    def __init__(self, popSize, crossoverRate, mutationRate):
        super().__init__()
        self.popSize = popSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.genomeLength = 0
        self.fittneses = [0]*self.popSize

    def prime(self, population, topography, loss):
        super().prime(population, topography, loss)
        self.genomeLength = sum(topography) - topography[-1]
        for i in range(0, self.popSize):
            super().initMember()

    def epochTrain(self, sample, classOf, learningRate=0.1):
        childPopulation = []
        for i in range(0, self.popSize):
            self.fittneses[i] = self.getFittness(i, [sample, classOf])
        while len(childPopulation) < self.popSize:
            parentIds = self.selection(2)
            children = self.crossover(parentIds)
            childPopulation.append(self.mutation(children[0]))
            childPopulation.append(self.mutation(children[1]))
        self.population = childPopulation

    def selection(self, selectCount, findBest=True):
        selectedIds = []
        current = -1
        rouletteSum = 0
        totalFitness = sum(self.fittneses)
        for i in range(0, selectCount):
            randPlace = random.random()
            while randPlace > rouletteSum:
                current += 1
                rouletteSum += (self.fittneses[current]/totalFitness) if findBest \
                    else (1-self.fittneses[current]/totalFitness)
            selectedIds.append(current)
        return selectedIds

    def crossover(self, parentIds):
        crossoverPoint = math.floor(self.genomeLength / self.crossoverRate * random.random())
        children = [[], []]
        for layer in self.population[parentIds[0]]:
            zeroes = np.zeros(layer.shape)
            children[0].append(zeroes)
            children[1].append(zeroes)
        parentOrder = [parentIds[0], parentIds[1]]
        i = 0
        j = 0
        index = 0
        while i < len(children[0]):
            children[0][i][j] = self.population[parentOrder[0]][i][j]
            children[1][i][j] = self.population[parentOrder[1]][i][j]
            j += 1
            index += 1
            if index == crossoverPoint:
                parentOrder = [parentIds[1], parentIds[0]]
            if j == children[0][i].shape[0]:
                j = 0
                i += 1
        return children

    def mutation(self, child):
        if random.random() > self.mutationRate:
            return child
        mutationPoint = math.floor(self.genomeLength * random.random())
        i = 0
        while mutationPoint >= child[i].shape[0]:
            mutationPoint -= child[i].shape[0]
            i += 1
        child[i][mutationPoint] = np.random.randn(child[i].shape[1])
        return child

    def getFittness(self, memberId, samples):
        self.fittneses[memberId] = 1 / self.loss(samples, memberId)[0]
        return self.fittneses[memberId]

    """flat = [self.population[parentIds[0]].ravel(), self.population[parentIds[1]].ravel()]
            parentOrder = [0, 1]
            i = 0
            j = 0
            index = 0
            while index < flat[0].size:
                children[0][i][j] = flat[parentOrder[0]][index]
                children[1][i][j] = flat[parentOrder[1]][index]
                j += 1
                index += 1
                if index == crossoverPoint:
                    parentOrder = [1, 0]
                if j == children[0][i].size:
                    j = 0
                    i += 1"""
