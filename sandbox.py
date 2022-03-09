# import the necessary packages
from NeuralNet import NeuralNet
import numpy as np
if __name__ == "__main__":
    # construct the XOR dataset
    sampleList = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    classList = np.array([[0], [1], [1], [0]])
    # define our 2-2-1 neural network and train it
    nn = NeuralNet([2, 2, 2])
    nn.train(sampleList, classList, epochs=20000, learningRate=0.5)
    # now that our network is trained, loop over the XOR data points
    for (sample, classOf) in zip(sampleList, classList):
        # make a prediction on the data point and display the result
        # to our console
        out = nn.test(sample)[0][0]
        step = 1 if out > 0.5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(sample, classOf[0], out, step))
