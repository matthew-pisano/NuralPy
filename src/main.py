import os
import time
import DecisionTree
from NeuralNet import NeuralNet
from train import *


def prepareData(fileName, normThreshDict, prune=0):
    """Import and trim data from CSV file"""
    # Trim least useful attribute
    DecisionTree.trimAttribDict(fileName, normThreshDict, prune)
    # Import total attribute and class data from CSV
    return Utils.importCSV(fileName, normThreshDict, "Diabetes")


def testProject(netConfig, trainingConfig):
    """Runs any supported configuration of the entire project"""
    # The modified diabetes file data
    fileName = os.path.dirname(os.path.realpath(__file__))+"/PrunedDiabetesDataSet.csv"
    # Dictionary containing the maximum and threshold values for each attribute
    # The threshold attribute decides weather an attribute is considered true or not
    normThreshDict = {
        "cholesterol": [500, 200],
        "glucose": [400, 140],
        "hdl_chol": [130, 60],
        "chol_hd_ratio": [30, 2],
        "age": [100, 40],
        "height": [80, 67],
        "weight": [350, 175],
        "gender": [1, 0.5],
        "bmi": [60, 30],
        "systolic_bp": [270, 120],
        "diastolic_bp": [150, 80]
    }
    # Prepare data from CSV
    trainSet, testSet = prepareData(fileName, normThreshDict, netConfig["prune"])
    # The number of nodes in each layer of the neural network
    netShape = [len(normThreshDict), 2]
    # Trainer passed into NeuralNet object determines what kind of algorithm will be used to train it
    if netConfig["type"] == "backprop":
        # For backpropogation
        net = NeuralNet(netShape, Backpropogator(netConfig["learningRate"]))
    else:
        # For the genetic algorithm
        net = NeuralNet(netShape, Genetic(netConfig["popSize"], netConfig["crossoverRate"], netConfig["mutationRate"]))
    t = time.time()
    # Train the neural network
    net.train(np.array(trainSet[0]), np.array(trainSet[1]), **trainingConfig)
    timeDiff = time.time() - t
    print("Trained after " + str(timeDiff) + "s")
    print("================================\n\n==============================")
    # Calculate loss of the trained neural network
    sampleSet = np.c_[np.array(testSet[0]), np.ones((np.array(testSet[0]).shape[0]))]
    loss = net.loss([sampleSet, np.array(testSet[1])], verbosity=3,
                    displaySamples=Utils.importCSV(fileName, normThreshDict, "Diabetes", normalize=False)[1])
    # Print out results
    print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")
    print("Overall score: " + str(loss[1] / timeDiff))


if __name__ == "__main__":
    backpropConfig = {
        "prune": 0,
        "type": "backprop",
        "learningRate": 0.18

    }
    geneticConfig = {
        "prune": 0,
        "type": "genetic",
        "popSize": 6,
        "crossoverRate": 0.5,
        "mutationRate": 0.01
    }
    trainingConfig = {
        "epochs": 150,
        "displayUpdate": 1,
        "verbosity": 1,
        "showPlots": True
    }
    # Run an example of the entire project
    # Use the backpropConfig dict to use a backpropagation network
    # and the geneticConfig dict to run the genetic algorithm network
    # trainingConfig contains parameters for actually training the network.  Variations in these values can
    # greatly effect the results!
    # The verbosity value determines how much detail will be included in the output
    # The showPlots value will show the python plots for the loss and accuracy curves of the network while it is training
    # The prune value determines the number of attributes to prune before execution
    testProject(backpropConfig, trainingConfig)
