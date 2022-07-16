import pandas as pd
import numpy as np

def confusion_matrix(pred: list, y_true: list, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) -> pd.DataFrame:
    """
    Returns the confusion matrix in a DataFrame for the given predictions and true values.
    """
    cm = pd.DataFrame(np.zeros((3, 3)), index = labels, columns = labels)
    for pred, true in zip(pred, y_true):
        cm.loc[true, pred] += 1

    return cm

def accuracy_score(pred: list, y_true: list) -> float:
    """
    Returns the accuracy score for the given predictions and true values.
    """
    true_positives = 0 
    for p, t in zip(pred, y_true):
        if p == t:
            true_positives += 1
    
    return true_positives / len(pred)

def precision_score(pred: list, y_true: list, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) -> pd.DataFrame:
    """
    Returns the precision score for the given predictions and true values.
    """
    cm = confusion_matrix(pred, y_true)
    precision = [0, 0, 0]
    for idx, label in enumerate(labels):
        precision[idx] = cm.loc[label, label] / np.sum(cm.loc[:, label])

    return pd.DataFrame(precision, index = labels)

def recall_score(pred: list, y_true: list, labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) -> pd.DataFrame:
    """
    Returns the recall score for the given predictions and true values.
    """
    cm = confusion_matrix(pred, y_true)
    recall = [0, 0, 0]
    for idx, label in enumerate(labels):
        recall[idx] = cm.loc[label, label] / np.sum(cm.loc[label, :])

    return pd.DataFrame(recall, index = labels)

def f1_score(pred: list, y_true: list) -> float:
    """
    Returns the f1 score for the given predictions and true values.
    """
    precision = precision_score(pred, y_true)
    recall = recall_score(pred, y_true)

    return 2 * (precision * recall) / (precision + recall)