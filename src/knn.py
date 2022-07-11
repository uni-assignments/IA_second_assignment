import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from main import read_data, divide_features_and_label, euclidean_distance

def most_frequent(List):
    return max(set(List), key = List.count)

def knn_predict_single(new_sample, x_train, y_train, k):

    heap = []
    for idx, features in x_train.iterrows():
        distance = euclidean_distance(new_sample, features)
        heapq.heappush(heap, (distance, y_train[idx]))

    nearest_neighboors = heapq.nsmallest(k, heap)

    return most_frequent([neighbor[1] for neighbor in nearest_neighboors])


def knn_predict_multiple(x_train, y_train, x_test, k):
    pred = []
    for idx, features in x_test.iterrows():
        pred.append(knn_predict_single(features, x_train, y_train, k))

    return pred

def knn_test(x_train, y_train, x_test, y_test, k = 8):
    
    pred = knn_predict_multiple(x_train, y_train, x_test, k)
    
    cm = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')

    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)


def sklearn_comparison(k = 8):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    
    return neigh.predict(x_test)


