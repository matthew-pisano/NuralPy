# import the necessary packages
from NeuralNet import NeuralNet
import cv2
import Utils
import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    normDict = {
        "cholesterol": 500,
        "glucose": 400,
        "hdl_chol": 130,
        "chol_hd_ratio": 30,
        "age": 100,
        "gender": 1,
        "bmi": 60,
        "systolic_bp": 270,
        "diastolic_bp": 150
    }
    samples, classes = Utils.importCSV("DiabetesDataSet.csv", normDict, "Diabetes")
    print(samples, classes)
