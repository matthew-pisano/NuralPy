import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.nio.file.*;

public class DecisionTree {
    private static class Node{
        public Node left;
        public Node right;
        public int attrib;
        public int classifier;
        public Node(){
            left = null;
            right = null;
            attrib = -1;
            classifier = -1;
        }
    }
    private static class SamplePoint{
        /**Values of the attributes of the sample point*/
        public double[] attributes;
        /**Classification of the sample point*/
        public double classifier;

        public SamplePoint(double classifier, double...attribs){
            attributes = attribs;
            this.classifier = classifier;
        }
        public String toString(){
            return "Class: "+classifier;
        }
    }
    /**Holds all the training sample points*/
    private static final ArrayList<SamplePoint> trainingSet = new ArrayList<>();
    /**Holds all the testing sample points*/
    private static final ArrayList<SamplePoint> testingSet = new ArrayList<>();
    /**List of thresholds of the attributes, results in binary split of each attribute*/
    private static double[] thresholds;
    /**Returns the entropy of the given set of samples for the given attribute
     * If the value of the attribIndex is -1, then the entropy for the overall set is calculated*/
    private static double calcEntropy(ArrayList<SamplePoint> decisionSamples, int attribIndex){
        // [[num attribClass1, num attribClass2, num class1 given attribClass1, num class1 given attribClass2]]
        int[][] attribFrequency = new int[decisionSamples.get(0).attributes.length + 1][4];
        // Loop for each of the samples
        for(SamplePoint sample : decisionSamples) {
            // For each of the attributes in each sample
            for (int j = 0; j < sample.attributes.length; j++) {
                if (sample.attributes[j] < thresholds[j]) {
                    // Increment the amount of the first class of attribute
                    attribFrequency[j][0]++;
                    if (sample.classifier < thresholds[thresholds.length - 1])
                        // If the sample is in the first class
                        attribFrequency[j][2]++;
                } else {
                    // Increment the amount of the second class of attribute
                    attribFrequency[j][1]++;
                    if (sample.classifier < thresholds[thresholds.length - 1])
                        // If the sample is in the first class
                        attribFrequency[j][3]++;
                }
            }
            if (sample.classifier < thresholds[thresholds.length-1])
                // Increment the total instances of the first class
                attribFrequency[attribFrequency.length-1][0]++;
            // Increment the total instances of the second class
            else attribFrequency[attribFrequency.length-1][1]++;
        }
        double class1 = attribFrequency[attribFrequency.length-1][0];
        double class2 = attribFrequency[attribFrequency.length-1][1];
        double samples = class1 + class2;
        // Calculate entropy for whole decision
        if(attribIndex == -1)
            return -class1/samples*Math.log(class1/samples)/Math.log(2)-class2/samples*Math.log(class2/samples)/Math.log(2);
        double attribC1 = attribFrequency[attribIndex][0];
        double attribC2 = attribFrequency[attribIndex][1];
        double attribC1Class1 = attribFrequency[attribIndex][2];
        double attribC2Class1 = attribFrequency[attribIndex][3];
        // Calculate entropy for specific attribute within decision
        double entropy = 0;
        if(attribC1Class1 != 0 && attribC1 != 0 && attribC1Class1 != attribC1)
            entropy += attribC1/samples*(-attribC1Class1/attribC1*Math.log(attribC1Class1/attribC1)/Math.log(2)-(1-attribC1Class1/attribC1)*Math.log(1-attribC1Class1/attribC1)/Math.log(2));
        if(attribC2Class1 != 0 && attribC2 != 0 && attribC2Class1 != attribC2)
            entropy += attribC2/samples*(-attribC2Class1/attribC2*Math.log(attribC2Class1/attribC2)/Math.log(2)-(1-attribC2Class1/attribC2)*Math.log(1-attribC2Class1/attribC2)/Math.log(2));
        return entropy;
    }
    /**Overloaded function for calculating entropy for entire set*/
    private static double calcEntropy(ArrayList<SamplePoint> decisionSamples){
        return calcEntropy(decisionSamples, -1);
    }
    /**Analyzes a set of sample points, whether the points all have the same class and the more frequent class of the points
     * is returned*/
    private static int[] analyzePoints(ArrayList<SamplePoint> decisionSamples){
        // If every sample has the same class
        int uniformClass = 1;
        int[] classFrequencies = new int[2];
        // Most frequent of the two classes
        int mostFrequent;
        for(int i=0; i< decisionSamples.size(); i++){
            // Update frequencies of classes
            if(decisionSamples.get(i).classifier == 0) classFrequencies[0] ++;
            else classFrequencies[1] ++;
            // Check if all samples have the same class
            if(uniformClass == 1 && i > 0 && decisionSamples.get(i).classifier != decisionSamples.get(i-1).classifier) uniformClass = 0;
        }
        if(classFrequencies[0] >= classFrequencies[1]) mostFrequent = 0;
        else mostFrequent = 1;

        return new int[]{uniformClass, mostFrequent};
    }
    /**Sets the thresholds for the attributes and the classifier*/
    private static void setThresholds(double...thresholds){
        DecisionTree.thresholds = thresholds;
    }
    /**Parses the data from a csv file into the list of sample points*/
    private static void parseData(String fileName) throws IOException {
        Scanner scan = new Scanner(new File(fileName));
        Path file = Paths.get(fileName);
        long setLength = Files.lines(file).count();
        // Throw out first line containing CSV headers
        scan.nextLine();
        while(scan.hasNextLine()){
            String[] sampleList = scan.nextLine().split(",");
            double[] parsedList = new double[sampleList.length];
            for(int i=0; i<sampleList.length; i++)
                parsedList[i] = Double.parseDouble(sampleList[i]);
            SamplePoint sample = new SamplePoint(parsedList[parsedList.length-1], Arrays.copyOfRange(parsedList, 0, parsedList.length-1));
            // Split total samples by 80% into the training set and 20% into the testing set
            if(trainingSet.size() < 0.8*setLength) trainingSet.add(sample);
            else testingSet.add(sample);
        }
    }
    /**Builds one node of the decision tree
     * attributes is an array of indices of attributes to check for decisions*/
    private static void buildTree(ArrayList<SamplePoint> decisionSamples, int[] attributes, Node root, double threshold){
        int[] analyzed = analyzePoints(decisionSamples);
        double[] entropies = new double[decisionSamples.get(0).attributes.length + 1];
        // If the class of the sample set is uniform or if there are no attributes to check
        if(analyzed[0] == 1 || attributes.length == 0)
            // Assign node as leaf and give it the more frequent class of the given sample set
            root.classifier = analyzed[1];
        else{
            // Calc entropies for overall decision and for all attributes in the array
            entropies[entropies.length-1] = calcEntropy(decisionSamples);
            for (int attribute : attributes)
                entropies[attribute] = calcEntropy(decisionSamples, attribute);
            // The index of the attribute that results in the maximum information gain
            double maxGain = Integer.MIN_VALUE;
            int maxAttribIndex = -1;
            for (int i=0; i<attributes.length; i++){
                double gain = entropies[entropies.length-1] - entropies[attributes[i]];
                if(gain > maxGain){
                    maxAttribIndex = i;
                    maxGain = gain;
                }
            }
            // If the max information gain is less than the threshold
            if(entropies[entropies.length-1] - entropies[attributes[maxAttribIndex]] < threshold)
                // Assign node as leaf and give it the more frequent class of the given sample set
                root.classifier = analyzed[1];
            else{
                // Assign the node's attribute to the maximum info gain attribute
                root.attrib = attributes[maxAttribIndex];
                ArrayList<SamplePoint> partition1 = new ArrayList<>();
                ArrayList<SamplePoint> partition2 = new ArrayList<>();
                // Split given sample points by the two categories of the maximum attribute
                for(SamplePoint sample : decisionSamples) {
                    if (sample.attributes[attributes[maxAttribIndex]] < thresholds[attributes[maxAttribIndex]])
                        // Samples with an attribute value less than that attributes threshold
                        partition1.add(sample);
                    else
                        // Samples with an attribute value more than that attributes threshold
                        partition2.add(sample);
                }
                // A new array of attributes without the maximum attribute
                int[] strippedAttribs = new int[attributes.length-1];
                // Whether to shift the index over after passing the attribute to remove
                int modifier = 0;
                // Copy old attributes array to new one without maximum attribute
                for(int i=0; i<attributes.length; i++) {
                    if (i != maxAttribIndex) strippedAttribs[i + modifier] = attributes[i];
                    else modifier = -1;
                }
                // If there are points in the first partition
                if(partition1.size() > 0){
                    // Give node left branch
                    root.left = new Node();
                    // Build on left branch
                    buildTree(partition1, strippedAttribs, root.left, threshold);
                }
                // If there are points in the second partition
                if(partition2.size() > 0){
                    // Give node right branch
                    root.right = new Node();
                    // Build on right branch
                    buildTree(partition2, strippedAttribs, root.right, threshold);
                }
            }
        }
    }
    /**Traverse given tree by a sample's attributes until a class node is reached*/
    private static int runTree(SamplePoint sample, Node root){
        // The current node of the tree
        Node nodeAt = root;
        // Whether the tree will go down the left or right branch
        boolean split = sample.attributes[nodeAt.attrib] < thresholds[nodeAt.attrib];
        // Loop until nodeAt is a leaf
        while(split && nodeAt.left != null || !split && nodeAt.right != null){
            // Re-assign nodeAt to right or left node
            nodeAt = split ? nodeAt.left : nodeAt.right;
            if(nodeAt.attrib == -1)
                return nodeAt.classifier;
            split = sample.attributes[nodeAt.attrib] < thresholds[nodeAt.attrib];
        }
        return nodeAt.classifier;
    }
    /**Calculate the accuracy of a given tree over a given sample set*/
    private static double calcAccuracy(ArrayList<SamplePoint> decisionSamples, Node root){
        double correctGuesses = 0;
        // Loop through all samples
        for(SamplePoint sample : decisionSamples) {
            // Class of result of tree
            int classifier = runTree(sample, root);
            // Mark as correct if the guess matches the given class
            if (classifier == sample.classifier) correctGuesses ++;
        }
        return correctGuesses / decisionSamples.size();
    }
    public static void main(String[] args) throws IOException {
        setThresholds(200, 140, 60, 5, 45, 0.5, 66, 150, 25, 120, 80, 0.5);
        parseData("/home/matthew/OneDrive/Documents/pgrm/AI/src/DiabetesDataSet.csv");
        Node root = new Node();
        // Build tree
        buildTree(trainingSet, new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, root, 0.005);
        double accuracy = calcAccuracy(testingSet, root);
        System.out.println("Accuracy of tree: "+accuracy);
    }
}
