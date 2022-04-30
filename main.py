import time
import DecisionTree
from NeuralNet import NeuralNet
from train import *


def prepareData(fileName, normThreshDict):
    # Trim least useful attribute
    DecisionTree.trimAttribDict(fileName, normThreshDict, 1)
    # Import total attribute and class data from CSV
    totalAttribs, totalClass = Utils.importCSV(fileName, normThreshDict, "Diabetes")
    # Training set, a tuple of attribute and class data
    trainSet = ([], [])
    # Testing set, a tuple of attribute and class data
    testSet = ([], [])
    i = 0
    # Divide sample points into training and testing sets
    while i < len(totalAttribs):
        if i < .8 * len(totalAttribs):
            trainSet[0].append(totalAttribs[i])
            trainSet[1].append(totalClass[i])
        else:
            testSet[0].append(totalAttribs[i])
            testSet[1].append(totalClass[i])
        i += 1
    return trainSet, testSet


if __name__ == "__main__":
    # The modified diabetes file data
    fileName = "PrunedDiabetesDataSet.csv"
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
    trainSet, testSet = prepareData(fileName, normThreshDict)
    # The number of nodes in each layer of the neural network
    netShape = [len(normThreshDict), 2]
    # For backpropogation
    net = NeuralNet(netShape, Backpropogator(learningRate=0.18))
    # For the genetic algorithm
    # net = NeuralNet(netShape, Genetic(6, 0.5, 0.01))
    t = time.time()
    # Train the neural network
    net.train(np.array(trainSet[0]), np.array(trainSet[1]), epochs=150, displayUpdate=1, verbosity=1, showPlots=False)
    timeDiff = time.time() - t
    print("Trained after " + str(timeDiff) + "s")
    print("================================\n\n==============================")
    # Calculate loss of the trained neural network
    sampleSet = np.c_[np.array(testSet[0]), np.ones((np.array(testSet[0]).shape[0]))]
    loss = net.loss([sampleSet, np.array(testSet[1])], verbosity=1)
    print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")
    print("Overall score: "+str(loss[1]/timeDiff))
