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
