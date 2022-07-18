# NeuralPy

A python research project to analyze the differences between backpropagation, a genetic algorithm method, and the decision tree optimization on the training and execution of a neural network.

## Aims and Objectives

The objective of the project is to compare the methods of backpropagation,
genetic algorithms, and decision trees on their abilities to predict a positive
diabetes diagnosis. For this project, we based our results on a training set
from the Vanderbilt University Department of Biostatistics.

## Implementation

The first steps to conducting the experiment were to craft each of the different
neural networks. The implementation of each network was conducted in
Python with extensive use of the *NumPy* library for its light and
efficient array and matrix implementations.

## Training and Testing

For training, each one of the three networks was given the same 80% proportion of the original data set and trained for 150 epochs and three independent trials. The loss function for each network is given by the sum of squared errors for every sample.

Each trial was given a score for comparison. The score is given by the percent of correctly identified diagnoses divided by the time it took for the trial to complete. The higher the score, the better the performance of the algorithm.

---

## Execution

**Requirements**:

- Python 3.8 or higher
- NumPy
- MatPlotLib
- Cv2

Running the main project file is as simple as:

> `python main.py`

when in the src directory of the project.
By default, this will run the project in the exact configuration that was used in the presentation slides.

Modifying any of the parameters in the dictionaries in the main method will result in different execution behavior.  After each successful execution, the program will output the results of the testing, such as which guesses were correct or incorrect.
The program will also display a plot showing the loss and accuracy over time.

The `DecisionTree.py` file contains the logic for the decision tree.  Actually running the decision tree is not utilized in the main project, so this can be tested in the main method here.  Through this file, the decision tree can be run and tested for its accuracy.

The `TreeNeuralNet.py` and `Neuron.py` files are deprecated.  They are not utilized anywhere in the program and only exist for showing how the project has evolved.
Their run configurations were replaced early on in the project.  The active utilization of this section of code can be accessed from early commits from the github repository.
