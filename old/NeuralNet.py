from Neuron import Neuron


class NeuralNet:
    def __init__(self, topography):
        self.network = [[]]*len(topography)
        self.network[0] = []
        for i in range(0, topography[0]):
            self.network[0].append(Neuron(i, 1))
        for i in range(1, len(topography)):
            self.network[i] = []
            for j in range(0, topography[i]):
                neuron = Neuron(i * 10000 + j, topography[i - 1])
                neuron.mapWeights(self.network[i-1])
                neuron.initialize()
                self.network[i].append(neuron)

        for i in range(0, len(self.network)-1):
            for n in self.network[i]:
                n.setConnections(self.network[i+1])

    def train(self, inputs, learningRate, goal, iterations):
        output = self.test(inputs)
        for itera in range(0, iterations+1):
            for outI in range(0, len(self.network[-1])):
                self.network[-1][outI].correctOutWeights(goal[outI] - output[outI], learningRate)
            for i in range(1, len(self.network)+1):
                for j in range(0, len(self.network[-i])):
                    self.network[-i][j].correctWeights(learningRate)

            output = self.test(inputs)
            # print("Test Output "+str(itera)+": " + str(output) + ", Desired output: " + str(goal))
        avgError = 0
        for i in range(0, len(output)):
            avgError += (output[i]-goal[i])/(goal[i]+1)
        avgError /= len(output)
        # System.out.println("Final Output: "+Arrays.toString(output)+", Desired output: "+Arrays.toString(goal))
        outMap = {"output": output, "error": avgError}
        return outMap

    def test(self, inputs):
        output = [0.0] * len(self.network[-1])
        for i in range(0, len(inputs)):
            self.network[0][i].getSignal(inputs[i], -1)
        for i in range(0, len(self.network[-1])):
            output[i] = self.network[-1][i].getSum()
        return output

    def saveWeights(self, outFile):
        saveStr = ""
        for layer in self.network:
            for neuron in layer:
                for weight in neuron.population.values():
                    saveStr += str(weight[0])+","
        with open(outFile, "w") as file:
            file.write(saveStr)

    def loadWeights(self, inFile):
        with open(inFile, "r") as file:
            saveStr = file.read()
        saveStr = saveStr.split(",")
        index = 0
        for layer in self.network:
            for neuron in layer:
                for weight in neuron.population.values():
                    weight[0] = float(saveStr[index])
                    index += 1
