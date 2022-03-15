# import the necessary packages
from NeuralNet import NeuralNet
import cv2
import matplotlib.pyplot as plt
import numpy as np


def scale(image, newDims):
    newImage = []
    xStep = image.shape[0] // newDims[0]
    yStep = image.shape[1] // newDims[1]
    prevX = 0
    prevY = 0
    xAt = 0
    yAt = 0
    average = 0
    pixelsCounted = 0
    color = 0
    while yAt < image.shape[1]:
        if xAt + 1 >= image.shape[0]:
            prevX = 0
            xAt = prevX
            prevY += yStep
            yAt = prevY
        if yAt == image.shape[1]:
            return newImage
        if xAt < prevX+xStep:
            average += image[yAt][xAt]
            if yAt == prevY+yStep and xAt == prevX+xStep:
                yAt -= yStep
                prevY = yAt
                prevX = xAt
                color = average / pixelsCounted
                # color = len(newImage) // 2
                # color += 0.5
                newImage.append(color if color < 255 else 255)
                # if color == 1:
                #    color = -0.5
                average = 0
                pixelsCounted = 0
            else:
                xAt += 1
                pixelsCounted += 1
        elif yAt < prevY+yStep:
            yAt += 1
            xAt = prevX

    return newImage


def toImage(array, size):
    imgArray = np.array([[0]*size]*size)
    for i in range(0, len(array)):
        imgArray[i // size][i % size] = array[i]
        if i == size**2 - 1 and i // size == size - 1:
            return imgArray
    return imgArray


def flatten(dataPoints):
    return [i for sub in dataPoints for i in sub]


if __name__ == "__main__":
    # construct the XOR dataset
    """sampleList = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
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
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(sample, classOf[0], out, step))"""

    img = cv2.imread("slash.jpg", cv2.IMREAD_GRAYSCALE)

    # newImg = scale(img, [300, 300])
    newImg = flatten(img)
    reconstruct = toImage(newImg, 300)
    plt.imshow(np.array(reconstruct), cmap="gray")
    plt.show()

    """h = img.shape[0]
    w = img.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            img[y][x] = float(img[y][x]) / 255.0
            if float(img[y][x]) != 255:
                print(float(img[y][x]) / 255.0)

    print("Showing image")
    plt.imshow(img, cmap="gray")
    plt.show()"""
