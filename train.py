import random
import numpy as np
import math
import Utils


class Trainer:
    """Parent class representing a training method for the neural network"""
    def __init__(self):
        self.population = None
        self.loss = None
        self.topography = None
        self.fittneses = [0]
        self.popSize = 1
        self.type = ""

    def prime(self, population, topography, loss):
        """Initialize additional properties"""
        self.population = population
        self.loss = loss
        self.topography = topography

    def selection(self, selectCount):
        """Selects the best `selectCount` number of networks from the population.
        Selects the first if there is only one in the population"""
        if len(self.population["pop"]) == 1:
            return [0]
        # Roulette wheel selection for multiple networks in population
        selectedIds = []
        totalFitness = sum(self.fittneses)
        while len(selectedIds) < selectCount:
            randPlace = random.random()
            rouletteSum = 0
            current = -1
            # Add to sum while greater than pointer
            while randPlace > rouletteSum:
                current += 1
                rouletteSum += self.fittneses[current]/totalFitness
            if current not in selectedIds:
                # Append to selected ids
                selectedIds.append(current)
        return selectedIds

    def getFitness(self, memberId, samples):
        """Returns the fittness of the given network over the given sample set"""
        return 1 / self.loss(samples, memberId)[0]

    def setAllFitness(self, samples):
        """Sets the fittness array to an array of fittnesses of the elements"""
        for i in range(0, self.popSize):
            self.fittneses[i] = self.getFitness(i, samples)

    def train(self, sampleList, classList, epochs=100, displayUpdate=10, verbosity=0, showPlots=False):
        """Trains the population of networks over the given sample set"""
        epochLosses = []
        epochAccuracy = []
        epochTimes = []
        # Append column to 1's to allow for training thresholds
        sampleList = np.c_[sampleList, np.ones((sampleList.shape[0]))]
        sampleList = np.atleast_2d(sampleList)
        classList = np.atleast_2d(classList)
        for epoch in range(0, epochs):
            # Train on each sample
            sampleNum = 0
            for (sample, classOf) in zip(sampleList, classList):
                self.epochTrain(sample, classOf)
                sampleNum += 1
            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                totalLoss = [0, 0]
                bestLoss = [float("infinity"), 0]
                mostCorrect = [0, 0]
                for i in range(0, len(self.population["pop"])):
                    loss = self.loss([sampleList, classList], i, verbosity)
                    totalLoss[0] += loss[0]
                    totalLoss[1] += loss[1]
                    if loss[0] < bestLoss[0]:
                        bestLoss = loss
                    if loss[1] > mostCorrect[1]:
                        mostCorrect = loss
                totalLoss[0] /= len(self.population["pop"])
                totalLoss[1] /= len(self.population["pop"])
                # Display network with the lowest loss and accuracy and the overall loss and accuracy for the population
                epochLosses.append(totalLoss[0])
                epochAccuracy.append(totalLoss[1])
                epochTimes.append(epoch)
                print("Epoch: " + str(epoch) + ", Average Loss: " + str(totalLoss[0]) + (
                            ", Correct: " + str(totalLoss[1] * 100) + "%" if verbosity > 0 else ""), end="")
                if len(self.population["pop"]) > 1:
                    print(", Best loss: "+str(bestLoss)+", most correct: "+str(mostCorrect))
                else:
                    print()
        # Show python plots
        if showPlots:
            algorithmName = "Backpropagation" if self.type == "backprop" else "Genetic Algorithm"
            Utils.plot(epochTimes, epochLosses, "Loss", algorithmName+" Loss Over Epochs")
            Utils.plot(epochTimes, epochAccuracy, "Accuracy", algorithmName+" Accuracy Over Epochs")

    def epochTrain(self, sample, classOf):
        """Stub method to be overriden by subclasses"""
        pass

    def initMember(self):
        """Initialize network and add to the population"""
        self.population["pop"].append([])
        # Initialize weights and thresholds for all but last layer
        for i in range(0, len(self.topography) - 2):
            weight = np.random.randn(self.topography[i] + 1, self.topography[i + 1] + 1)
            self.population["pop"][-1].append(weight / np.sqrt(self.topography[i]))
        # Initialize weights and thresholds for output layer
        weight = np.random.randn(self.topography[-2] + 1, self.topography[-1])
        self.population["pop"][-1].append(weight / np.sqrt(self.topography[-2]))


class Backpropogator(Trainer):
    """Trainer method for backpropogation"""
    def __init__(self, learningRate):
        super().__init__()
        self.learningRate = learningRate
        self.type = "backprop"

    def prime(self, population, topography, loss):
        """Initialize this network"""
        super().prime(population, topography, loss)
        super().initMember()

    def epochTrain(self, sample, classOf):
        """Train this network for one epoch"""
        activations = [np.asarray([sample])]
        # Gather activations for each layer
        for layer in range(0, len(self.population["pop"][0])):
            activation = activations[layer].dot(self.population["pop"][0][layer])
            activations.append(Utils.sigmoid(activation))
        # Calculate error of output layer
        error = activations[-1] - classOf
        deltas = [error * Utils.sigmoidDx(activations[-1])]
        # Calculate change in weights
        for layer in range(len(activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.population["pop"][0][layer].transpose())
            delta = delta * Utils.sigmoidDx(activations[layer])
            deltas.append(delta)
        # Update weights
        for layer in range(0, len(self.population["pop"][0])):
            self.population["pop"][0][layer] += -self.learningRate * activations[layer].transpose().dot(deltas[-(layer + 1)])


class Genetic(Trainer):
    """Trainer method for the genetic algorithm"""
    def __init__(self, popSize, crossoverRate, mutationRate):
        super().__init__()
        self.popSize = popSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.genomeLength = 0
        self.type = "genetic"

    def prime(self, population, topography, loss):
        """Initialize the genome length and create the fittness array"""
        super().prime(population, topography, loss)
        self.genomeLength = sum(topography) - topography[-1]
        self.fittneses = [0] * self.popSize
        # Initialize all members
        for i in range(0, self.popSize):
            super().initMember()

    def epochTrain(self, sample, classOf):
        """Breed best parents until entire population has been replaced with next generation"""
        childPopulation = []
        self.setAllFitness([sample, classOf])
        while len(childPopulation) < self.popSize:
            parentIds = self.selection(2)
            children = self.crossover(parentIds)
            childPopulation.append(self.mutation(children[0]))
            childPopulation.append(self.mutation(children[1]))
        # Replace current population with new population
        self.population["pop"] = childPopulation

    def crossover(self, parentIds):
        """Breed parents to create children using a single point crossover"""
        crossoverPoint = math.floor(self.genomeLength / self.crossoverRate * random.random())
        # Child genomes
        children = [[], []]
        for layer in self.population["pop"][parentIds[0]]:
            zeroes = np.zeros(layer.shape)
            children[0].append(zeroes)
            children[1].append(zeroes)
        i = 0
        # Fill first half of genome before crossover
        while children[0][i].shape[0] < crossoverPoint:
            children[0][i] = self.population["pop"][parentIds[0]][i]
            children[1][i] = self.population["pop"][parentIds[1]][i]
            crossoverPoint -= children[0][i].shape[0]
            i += 1
            if i >= len(children[0]):
                return children
        # Fill subsections of array was bisected by crossover
        children[0][i][:crossoverPoint] = self.population["pop"][parentIds[0]][i][:crossoverPoint]
        children[1][i][:crossoverPoint] = self.population["pop"][parentIds[1]][i][:crossoverPoint]
        children[0][i][crossoverPoint:] = self.population["pop"][parentIds[1]][i][crossoverPoint:]
        children[1][i][crossoverPoint:] = self.population["pop"][parentIds[0]][i][crossoverPoint:]
        # Fill remaining genome with opposite parent
        i += 1
        while i < len(children[0]):
            children[0][i] = self.population["pop"][parentIds[1]][i]
            children[1][i] = self.population["pop"][parentIds[0]][i]
            i += 1
        return children

    def mutation(self, child):
        """Preform mutation on the given genome"""
        if random.random() > self.mutationRate:
            return child
        mutationPoint = math.floor(self.genomeLength * random.random())
        i = 0
        # Loop to mutation point
        while mutationPoint >= child[i].shape[0]:
            mutationPoint -= child[i].shape[0]
            i += 1
        # Randomly edit mutation point
        child[i][mutationPoint] = np.random.randn(child[i].shape[1])
        return child
