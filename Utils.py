import math
import numpy as np
import csv
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoidDx(x):
    return x * (1 - x)


def tanh(x):
    try:
        math.pow(math.e, -0.667 * x)
    except Exception as e:
        print(e)
    return 3.432/(1+math.pow(math.e, -0.667*x)) - 1.716


def toImage(array, size):
    imgArray = np.array([[0.0]*size]*size)
    for i in range(0, len(array)):
        imgArray[i // size][i % size] = array[i]
        if i == size**2 - 1 and i // size == size - 1:
            return imgArray
    return imgArray


# Use dict to store maximum values for normalization
def importCSV(fileName, normDict, classifier):
    samples = []
    classes = []
    with open(fileName, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rowList = []
            # Normalize data and add to list
            for key, value in normDict.items():
                if row[key].count("male") > 0:
                    rowList.append(0 if row[key] == "male" else 1)
                else:
                    rowList.append(float(row[key])/value)
            # Check for 'o' in 'No diabetes'
            classes.append([1, 0] if row[classifier].count("o") > 0 else [0, 1])
            samples.append(rowList)
    return np.asarray(samples), np.asarray(classes)


def plot(xPoints, yPoints, yLabel, title):
    plt.scatter(xPoints, yPoints)
    plt.plot(xPoints, yPoints)
    plt.xlabel("Epoch")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()




