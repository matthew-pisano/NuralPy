import os.path
import time
import cv2
import numpy as np
from NeuralNet import NeuralNet
import tensorflow as tf

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
    """dateset = tf.keras.datasets.mnist
    (imgTrain, classTrain), (imgTest, classTest) = dateset.load_data()

    imgTrain = tf.keras.utils.normalize(imgTrain, axis=1)
    imgTest = tf.keras.utils.normalize(imgTest, axis=1)"""
    normDict = {
        "cholesterol": 500,
        "glucose": 400,
        "hdl_chol": 130,
        # "chol_hd_ratio": 30,
        "age": 100,
        "gender": 1,
        "bmi": 60,
        "systolic_bp": 270,
        "diastolic_bp": 150
    }
    imgTrain, classTrain = Utils.importCSV("DiabetesDataSet.csv", normDict, "Diabetes")
    trainTest = True, True
    trainSet = ([], [])
    testSet = ([], [])
    testLim = 380
    trainLim = testLim * .8
    tests = 0
    # netShape = [784, 128, 128, 10]
    netShape = [8, 5, 2]
    net = NeuralNet(netShape, Backpropogator(learningRate=0.5))
    # net = NeuralNet(netShape, Genetic(10, 0.8, 0.1))
    """if saveResume and os.path.exists("weights.csv"):
        net.loadWeights("weights.csv")"""
    while tests < testLim:
        # rawTrain = flatten(toTuple(imgTrain[tests]))
        rawTrain = imgTrain[tests]
        rawClass = classTrain[tests]
        """rawClass = [0] * classTrain[0].shape[0]
        for j in range(0, classTrain[0].shape[0]):
            if j == classTrain[tests]:
                rawClass[j] = 1"""
        if tests < trainLim:
            trainSet[0].append(rawTrain)
            trainSet[1].append(rawClass)
        else:
            testSet[0].append(rawTrain)
            testSet[1].append(rawClass)
        tests += 1
    tests = 0
    if trainTest[0]:
        t = time.time()
        out = net.train(np.array(trainSet[0]), np.array(trainSet[1]), epochs=100, displayUpdate=1, verbosity=1)
        timeDiff = time.time() - t
        print("Trained after " + str(timeDiff) + "s")
        print("================================\n\n==============================")
        net.saveWeights("save.w")
    if trainTest[1]:
        """try:
            net.loadWeights("save.w")
        except FileNotFoundError as e:
            print("Could not load weights file")"""
        sampleSet = np.c_[np.array(testSet[0]), np.ones((np.array(testSet[0]).shape[0]))]
        loss = net.loss([sampleSet, np.array(testSet[1])], verbosity=1)
        print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")
        print("Overall score: "+str(loss[1]/timeDiff))
        """img = cv2.imread("zero.jpg", cv2.IMREAD_GRAYSCALE)
        newImg = normalize(flatten(img))
        sampleSet = np.c_[np.array([newImg]), np.ones((np.array([newImg]).shape[0]))]
        loss = net.loss([sampleSet, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])], verbosity=1)
        print("Loss: " + str(loss[0]) + ", Correct: " + str(loss[1] * 100) + "%")"""
