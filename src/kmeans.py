import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pip import main

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from main import read_data, divide_features_and_label, euclidean_distance



def kmeans_fit(x_train):



if __name__ == '__main__':

    train_data_path = "../iris treino.csv"
    train_data = read_data(train_data_path)

    test_data_path = "../iris aleatório.csv"
    test_data = read_data(test_data_path)

    x_test, y_test = divide_features_and_label(test_data) 
    x_train, y_train = divide_features_and_label(train_data) 