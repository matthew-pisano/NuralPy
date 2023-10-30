import numpy as np


class ActivationFn:
    """An activation function base class to inherit from"""

    @classmethod
    def forward(cls, x):
        """The activation function"""

        ...

    @classmethod
    def backward(cls, a_dx, x):
        """The derivative of the activation function for the backward pass"""

        ...


class LossFn:
    """A loss function base class to inherit from"""

    def __call__(self, output: np.ndarray, target: np.ndarray):
        ...


class CrossEntropy(LossFn):
    """A cross entropy loss function"""

    def __call__(self, output: np.ndarray, target: np.ndarray, epsilon=1e-12):
        output = np.clip(output, epsilon, 1. - epsilon)
        shape = output.shape[1]
        ce = -np.sum(target * np.log(output + 1e-9)) / shape
        return ce


class Sigmoid(ActivationFn):
    """A Sigmoid activation function"""

    @classmethod
    def forward(cls, x):
        return 1.0 / (1 + np.exp(-x))

    @classmethod
    def backward(cls, a_dx, x):
        sig = cls.forward(x)
        return a_dx * sig * (1 - sig)


class ReLU(ActivationFn):
    """A ReLU activation function"""

    @classmethod
    def forward(cls, x):
        return np.maximum(0, x)

    @classmethod
    def backward(cls, a_dx, x):
        x_dx = np.array(a_dx, copy=True)
        x_dx[x <= 0] = 0
        return x_dx
