import numpy as np
from keras.datasets import mnist

from network import Network, Layer
from utils import ReLU, Sigmoid, CrossEntropy


if __name__ == "__main__":

    # Load MNIST from keras
    (x_train, y_train_tmp), (x_test, y_test_tmp) = mnist.load_data()
    # Expand dimension of labels to 10
    y_train = np.zeros((y_train_tmp.shape[0], 10))
    y_test = np.zeros((y_test_tmp.shape[0], 10))
    for i in range(len(y_train_tmp)):
        y_train[i][y_train_tmp[i]] = 1
    for i in range(len(y_test_tmp)):
        y_test[i][y_test_tmp[i]] = 1

    # Normalize image data
    x_train = x_train.reshape(x_train.shape[0], 784)/255
    x_test = x_test.reshape(x_test.shape[0], 784)/255

    # Create layers and network
    topology = [784, 32, 10]
    neurotransmitters = 3
    layers = [Layer((topology[i], topology[i+1]), ReLU(), neurotransmitters=neurotransmitters) for i in range(len(topology)-2)]
    layers.append(Layer((topology[-2], topology[-1]), Sigmoid(), neurotransmitters=neurotransmitters))
    network = Network(layers, CrossEntropy())

    # Train network
    print(network.train(x_train, y_train, 10, learning_rate=3e-2))

    # Get sample output and print
    output = network.forward(np.atleast_2d(x_test[0]).T)
    print("Output:", output, ", Target:", y_test[0])
