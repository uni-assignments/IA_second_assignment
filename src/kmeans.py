import heapq, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pip import main
from collections import defaultdict, Counter
from typing import List

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from utils import read_data, divide_features_and_label, euclidean_distance


def get_initial_k_centroids(x_train: pd.DataFrame, k) -> List[List[float]]:
    """
    Returns k random points from the training set as the initial centroids.
    """
    possible_centroids = list(range(0, len(x_train)))
    random.shuffle(possible_centroids)
    centroids = possible_centroids[:k]

    return [x_train.loc[centroid].tolist() for centroid in centroids]

def get_new_centroids(x_train: pd.DataFrame, clusters: defaultdict(List[int])) -> List[List[float]]:
    """
    Returns the new centroids based on the mean of the points in each cluster.
    """
    new_centroids = []
    for centroid, elements in clusters.items():
        mean = np.sum(x_train.loc[elements])/len(elements)  
        new_centroids.append(mean.tolist())      

    return new_centroids

def cluster_samples(x_train: pd.DataFrame, centroids: List[List[float]]) -> defaultdict(List[int]):
    """
    Returns a dictionary of clusters, where the key is the centroid and the value is a list of samples that are closest to that centroid
    """
    centroid_members = defaultdict(list)
    for sample_idx, sample in x_train.iterrows():
        nearest_centroid_idx = min([(euclidean_distance(centroid, sample), idx) for idx, centroid in enumerate(centroids)])[1]
        centroid_members[nearest_centroid_idx].append(sample_idx)

    return centroid_members

def kmeans_fit(x_train: pd.DataFrame, k = 3) -> (List[List[float]], defaultdict(List[int])):
    """
    Returns the list with the centroid features and the clusters of the samples, where the key is the index of the centroid.
    """
    centroids = get_initial_k_centroids(x_train, k)
    centroid_members = cluster_samples(x_train, centroids)

    while True:
        new_centroids = get_new_centroids(x_train, centroid_members)
        new_centroid_members = cluster_samples(x_train, new_centroids)
        
        if new_centroid_members == centroid_members:
            break
        centroid_members = new_centroid_members

    return new_centroids, new_centroid_members

def analysis(x_train, y_train, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], k = 3) -> None:
    """
    Makes a comparison between the sample labels and the returned clusters.
    """
    centroids, centroid_members = kmeans_fit(x_train, k)

    cm = pd.DataFrame(np.zeros((k, 3)), index = list(range(0, k)), columns = labels)
    for centroid, elements in centroid_members.items():
        for element in elements:
            cm.loc[centroid, y_train[element]] += 1
    
    print(cm)

if __name__ == '__main__':

    train_data_path = "../iris treino.csv"
    train_data = read_data(train_data_path)

    test_data_path = "../iris aleatório.csv"
    test_data = read_data(test_data_path)

    x_test, y_test = divide_features_and_label(test_data) 
    x_train, y_train = divide_features_and_label(train_data) 

    # kmeans_fit(x_train)
    analysis(x_test, y_test, k = 3)
