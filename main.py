import os.path
import time
import cv2
import numpy as np

import DecisionTree
from NeuralNet import NeuralNet
# import tensorflow as tf

from train import *


def flatten(dataPoints):
    return [i for sub in dataPoints for i in sub]


def normalize(dataPoints):
    for x in range(0, len(dataPoints)):
        dataPoints[x] = float(dataPoints[x]) / 255.0
    return dataPoints


def toTuple(a):
    try:
        return tuple(toTuple(i) for i in a)
    except TypeError:
        return a


if __name__ == "__main__":
    fileName = "PrunedDiabetesDataSet.csv"
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
    DecisionTree.trimAttribDict(fileName, normThreshDict, 2)
    imgTrain, classTrain = Utils.importCSV(fileName, normThreshDict, "Diabetes")
    trainTest = True, True
    trainSet = ([], [])
    testSet = ([], [])
    testLim = 390
    trainLim = testLim * .8
    tests = 0
    netShape = [len(normThreshDict), 2]
    net = NeuralNet(netShape, Backpropogator(learningRate=0.18))
    # net = NeuralNet(netShape, Genetic(6, 0.5, 0.01))
    while tests < testLim:
        rawTrain = imgTrain[tests]
        rawClass = classTrain[tests]
        if tests < trainLim:
            trainSet[0].append(rawTrain)
            trainSet[1].append(rawClass)
        else:
            testSet[0].append(rawTrain)
            testSet[1].append(rawClass)
        tests += 1
    tests = 0
    timeDiff = None
    if trainTest[0]:
        t = time.time()
        out = net.train(np.array(trainSet[0]), np.array(trainSet[1]), epochs=150, displayUpdate=1, verbosity=1, showPlots=False)
        timeDiff = time.time() - t
        print("Trained after " + str(timeDiff) + "s")
        print("================================\n\n==============================")
    if trainTest[1]:
        sampleSet = np.c_[np.array(testSet[0]), np.ones((np.array(testSet[0]).shape[0]))]
        loss = net.loss([sampleSet, np.array(testSet[1])], verbosity=1)
        print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")
        if timeDiff:
            print("Overall score: "+str(loss[1]/timeDiff))
