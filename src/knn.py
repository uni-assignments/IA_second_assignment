import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from utils import read_data, divide_features_and_label, euclidean_distance
from metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def weighted_most_frequent(pred: List[str], k, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) -> str:
    """
    Returns the most frequent element in a list making a weighted mean, closest neighbors are more valuable. 
    Used to find the most frequent label in the k closest neighbors.
    """
    d = dict()
    for label in labels:
        d[label] = 0

    for idx, p in enumerate(pred):
        d[p] += (1 + (k - idx) * 1/k)/2

    return heapq.nlargest(1, d, key=d.get)[0]

def knn_predict_single(new_sample: pd.Series, x_train: pd.DataFrame, y_train: pd.DataFrame, k) -> str:
    """
    Returns the prediction for a single sample.
    """
    heap = []
    for idx, features in x_train.iterrows():
        distance = euclidean_distance(new_sample, features)
        heapq.heappush(heap, (distance, y_train[idx]))

    nearest_neighboors = heapq.nsmallest(k, heap)

    return weighted_most_frequent([neighbor[1] for neighbor in nearest_neighboors], k)

def knn_predict_multiple(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, k) -> List[str]: 
    """
    Returns a list of predictions for the test set.
    """
    pred = []
    for idx, features in x_test.iterrows():
        pred.append(knn_predict_single(features, x_train, y_train, k))

    return pred

def knn_test(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, k = 8) -> None:
    """
    Prints the accuracy, precision, recall and f1 score of the implemented knn algorithm for the dataset.
    """
    pred = knn_predict_multiple(x_train, y_train, x_test, k)
    print("Confusion Matrix:")
    print(confusion_matrix(pred, y_test))
    print("\nAccuracy:")
    print(accuracy_score(pred, y_test))
    print("\nPrecision:")
    print(precision_score(pred, y_test))
    print("\nRecall:")
    print(recall_score(pred, y_test))
    print("\nF1 Score:")
    print(f1_score(pred, y_test))

if __name__ == '__main__':

    train_data_path = "../iris treino.csv"
    train_data = read_data(train_data_path)

    test_data_path = "../iris aleatório.csv"
    test_data = read_data(test_data_path)

    x_test, y_test = divide_features_and_label(test_data) 
    x_train, y_train = divide_features_and_label(train_data) 

    knn_test(x_train, y_train, x_test, y_test, k = 8)
