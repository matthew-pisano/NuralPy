import math
import numpy as np


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




