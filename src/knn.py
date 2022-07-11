import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

def read_data(path):
    return pd.read_csv(path, sep=";", header = None, index_col = False)

def divide_features_and_label(train_data):
    return train_data.iloc[:, :-1], train_data.iloc[:, -1].tolist()

def euclidean_distance(new_sample, classified_element):

    squares = []
    for new_sample_feature, classified_feature in zip(new_sample, classified_element):
        squares.append(np.square(new_sample_feature - classified_feature))

    return np.sqrt(np.sum(squares))

def most_frequent(List):
    return max(set(List), key = List.count)

def knn_classification(new_sample, x_train, y_train, k = 8):

    heap = []
    for idx, features in x_train.iterrows():
        distance = euclidean_distance(new_sample, features)
        heapq.heappush(heap, (distance, y_train[idx]))

    nearest_neighboors = heapq.nsmallest(k, heap)

    return most_frequent([neighbor[1] for neighbor in nearest_neighboors])

def test(x_train, y_train, x_test, y_test, k = 8):

    pred = []
    for idx, features in x_test.iterrows():
        pred.append(knn_classification(features, x_train, y_train, k = 8))

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

if __name__ == '__main__':

    train_data_path = "../iris treino.csv"
    train_data = read_data(train_data_path)

    test_data_path = "../iris aleatório.csv"
    test_data = read_data(test_data_path)

    x_test, y_test = divide_features_and_label(test_data) 
    x_train, y_train = divide_features_and_label(train_data) 

    # test(test_data, train_data, k = 2)
    test(x_train, y_train, x_test, y_test, k = 8)
    # test(test_data, train_data, k = 32)
