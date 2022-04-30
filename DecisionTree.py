import math

trainingSet = []
"""Holds all the training sample points"""
testingSet = []
"""Holds all the testing sample points"""
thresholds = []
"""List of thresholds of the attributes, results in binary split of each attribute"""
attribGains = {}
titles = []


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attrib = -1
        self.classifier = -1


class SamplePoint:
    """Class representing a sample point of data"""
    def __init__(self, classifier, attribs):
        self.attributes = attribs
        """Values of the attributes of the sample point"""
        self.classifier = classifier
        """Classification of the sample point"""

    def __str__(self):
        return "Point Class: " + self.classifier


def calcEntropy(decisionSamples, useAttribs, attribIndex=None):
    """Returns the entropy of the given set of samples for the given attribute
     * If the value of the attribIndex is -1, then the entropy for the overall set is calculated"""
    if not attribIndex:
        return calcEntropy(decisionSamples, useAttribs, -1)
    attribFrequency = [[0, 0, 0, 0]] * (len(decisionSamples[0].attributes) + 1)
    # Calculate stats for each attribute for calculation
    # Loop for each of the samples
    for sample in decisionSamples:
        # For each of the attributes in each sample
        for j in range(len(useAttribs)):
            if sample.attributes[useAttribs[j]] < thresholds[useAttribs[j]]:
                # Increment the amount of the first class of attribute
                attribFrequency[j][0] += 1
                if sample.classifier < thresholds[-1]:
                    # If the sample is in the first class
                    attribFrequency[j][2] += 1
            else:
                # Increment the amount of the second class of attribute
                attribFrequency[j][1] += 1
                if sample.classifier < thresholds[-1]:
                    # If the sample is in the first class
                    attribFrequency[j][3] += 1
        if sample.classifier < thresholds[-1]:
            # Increment the total instances of the first class
            attribFrequency[-1][0] += 1
        # Increment the total instances of the second class
        else:
            attribFrequency[-1][1] += 1

    class1 = attribFrequency[-1][0]
    class2 = attribFrequency[-1][1]
    samples = class1 + class2
    # Calculate entropy for whole decision
    if attribIndex == -1:
        return -class1 / samples * math.log(class1 / samples, 2) - class2 / samples * math.log(class2 / samples, 2)
    attribC1 = attribFrequency[attribIndex][0]
    attribC2 = attribFrequency[attribIndex][1]
    attribC1Class1 = attribFrequency[attribIndex][2]
    attribC2Class1 = attribFrequency[attribIndex][3]
    # Calculate entropy for specific attribute within decision
    entropy = 0
    if attribC1Class1 != 0 and attribC1 != 0 and attribC1Class1 != attribC1:
        entropy += attribC1 / samples * (-attribC1Class1 / attribC1 * math.log(attribC1Class1 / attribC1, 2) - (
                    1 - attribC1Class1 / attribC1) * math.log(1 - attribC1Class1 / attribC1, 2))
    if attribC2Class1 != 0 and attribC2 != 0 and attribC2Class1 != attribC2:
        entropy += attribC2 / samples * (-attribC2Class1 / attribC2 * math.log(attribC2Class1 / attribC2, 2) - (
                    1 - attribC2Class1 / attribC2) * math.log(1 - attribC2Class1 / attribC2, 2))
    return entropy


def analyzePoints(decisionSamples):
    """Analyzes a set of sample points, whether the points all have the same class and the more frequent class of the points
     * is returned"""
    # If every sample has the same class
    uniformClass = 1
    classFrequencies = [0, 0]
    # Most frequent of the two classes
    mostFrequent = None
    for i in range(0, len(decisionSamples)):
        # Update frequencies of classes
        if decisionSamples[i].classifier == 0:
            classFrequencies[0] += 1
        else:
            classFrequencies[1] += 1
        # Check if all samples have the same class
        if uniformClass == 1 and i > 0 and decisionSamples[i].classifier != decisionSamples[i - 1].classifier:
            uniformClass = 0

    if classFrequencies[0] >= classFrequencies[1]:
        mostFrequent = 0
    else:
        mostFrequent = 1

    return uniformClass, mostFrequent


def setUseAttribs(attribDict):
    """Sets the thresholds for the attributes and the classifier"""
    global thresholds
    useAttribs = []
    for i in range(0, len(titles)):
        thresholds.append(0)
        if titles[i] in attribDict:
            thresholds[-1] = attribDict[titles[i]][1]
            useAttribs.append(i)
    thresholds.append(.5)
    return useAttribs


def parseData(fileName):
    """Parses the data from a csv file into the list of sample points"""
    global titles
    with open(fileName, "r") as file:
        lines = file.read().split("\n")
        # Throw out first line containing CSV headers
        titles = lines.pop(0).split(",")[:-1]
        setLength = len(lines)

        while len(lines) > 0:
            sampleList = lines.pop(0).split(",")
            if sampleList[0] == "":
                break
            parsedList = []
            for sample in sampleList:
                parsedList.append(float(sample))
            sample = SamplePoint(parsedList[-1], parsedList[:-1])
            # Split total samples by 80% into the training set and 20% into the testing set
            if len(trainingSet) < 0.8 * setLength:
                trainingSet.append(sample)
            else:
                testingSet.append(sample)


def buildTree(decisionSamples, attributes, root, threshold):
    """Builds one node of the decision tree
     * attributes is an array of indices of attributes to check for decisions"""
    analyzed = analyzePoints(decisionSamples)
    entropies = [0] * (len(decisionSamples[0].attributes) + 1)
    # If the class of the sample set is uniform or if there are no attributes to check
    if analyzed[0] == 1 or len(attributes) == 0:
        # Assign node as leaf and give it the more frequent class of the given sample set
        root.classifier = analyzed[1]
    else:
        # Calc entropies for overall decision and for all attributes in the array
        entropies[-1] = calcEntropy(decisionSamples, attributes)
        for attribute in attributes:
            entropies[attribute] = calcEntropy(decisionSamples, attributes, attribute)
        # The index of the attribute that results in the maximum information gain
        maxGain = -float("infinity")
        maxAttribIndex = -1
        for i in range(0, len(attributes)):
            gain = entropies[-1] - entropies[attributes[i]]
            if gain > maxGain:
                maxAttribIndex = i
                maxGain = gain

        # If the max information gain is less than the threshold
        if entropies[-1] - entropies[attributes[maxAttribIndex]] < threshold:
            # Assign node as leaf and give it the more frequent class of the given sample set
            root.classifier = analyzed[1]
        else:
            attribGains[attributes[maxAttribIndex]] = maxGain
            # Assign the node's attribute to the maximum info gain attribute
            root.attrib = attributes[maxAttribIndex]
            partition1 = []
            partition2 = []
            # Split given sample points by the two categories of the maximum attribute
            for sample in decisionSamples:
                if sample.attributes[attributes[maxAttribIndex]] < thresholds[attributes[maxAttribIndex]]:
                    # Samples with an attribute value less than that attributes threshold
                    partition1.append(sample)
                else:
                    # Samples with an attribute value more than that attributes threshold
                    partition2.append(sample)

            # A new array of attributes without the maximum attribute
            strippedAttribs = [0] * (len(attributes) - 1)
            # Whether to shift the index over after passing the attribute to remove
            modifier = 0
            # Copy old attributes array to new one without maximum attribute
            for i in range(0, len(attributes)):
                if i != maxAttribIndex:
                    strippedAttribs[i + modifier] = attributes[i]
                else:
                    modifier = -1

            # If there are points in the first partition
            if len(partition1) > 0:
                # Give node left branch
                root.left = Node()
                # Build on left branch
                buildTree(partition1, strippedAttribs, root.left, threshold)

            # If there are points in the second partition
            if len(partition2) > 0:
                # Give node right branch
                root.right = Node()
                # Build on right branch
                buildTree(partition2, strippedAttribs, root.right, threshold)


def runTree(sample, root):
    """Traverse given tree by a sample's attributes until a class node is reached"""
    # The current node of the tree
    nodeAt = root
    # Whether the tree will go down the left or right branch
    split = sample.attributes[nodeAt.attrib] < thresholds[nodeAt.attrib]
    # Loop until nodeAt is a leaf
    while split and nodeAt.left is not None or not split and nodeAt.right is not None:
        # Re-assign nodeAt to right or left node
        nodeAt = nodeAt.left if split else nodeAt.right
        if nodeAt.attrib == -1:
            return nodeAt.classifier
        split = sample.attributes[nodeAt.attrib] < thresholds[nodeAt.attrib]

    return nodeAt.classifier


def calcAccuracy(decisionSamples, root):
    """Calculate the accuracy of a given tree over a given sample set"""
    correctGuesses = 0
    # Loop through all samples
    for sample in decisionSamples:
        # Class of result of tree
        classifier = runTree(sample, root)
        # Mark as correct if the guess matches the given class
        if classifier == sample.classifier:
            correctGuesses += 1

    return correctGuesses / len(decisionSamples)


def trimAttribDict(fileName, attribDict, numTrim):
    parseData(fileName)
    useAttribs = setUseAttribs(attribDict)
    root = Node()
    # Build tree
    buildTree(trainingSet, useAttribs, root, 0.005)
    orderedAttribList = list(dict(sorted(attribGains.items(), key=lambda item: item[1])))
    for i in orderedAttribList[-numTrim:]:
        attribDict.pop(titles[i])


def main():
    attribDict = {
        "cholesterol": [500, 200],
        "glucose": [400, 140],
        "hdl_chol": [130, 60],
        "chol_hd_ratio": [30, 2],
        "age": [100, 40],
        "gender": [1, 0.5],
        "bmi": [60, 30],
        "systolic_bp": [270, 120],
        "diastolic_bp": [150, 80]
    }
    trimAttribDict("PrunedDiabetesDataSet.csv", attribDict, 3)
    print(attribDict)
    """parseData("DiabetesDataSet.csv")
    root = Node()
    # Build tree
    buildTree(trainingSet, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], root, 0.005)
    accuracy = calcAccuracy(testingSet, root)
    print("Accuracy of tree: " + str(accuracy))
    print("Attribute Gains: "+str(attribGains))"""


if __name__ == "__main__":
    main()
