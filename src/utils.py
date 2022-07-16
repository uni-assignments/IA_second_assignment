import pandas as pd
import numpy as np

def read_data(path: str) -> pd.DataFrame:
    """
    Reads the csv file in the path passed as param and returns a pandas dataframe.
    """
    return pd.read_csv(path, sep=";", header = None, index_col = False)

def divide_features_and_label(train_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Divides the received dataframe into two dataframes, one with the features and one with the labels. 
    Returns both in a tuple.
    """
    return train_data.iloc[:, :-1], train_data.iloc[:, -1].tolist()

def euclidean_distance(point1: pd.Series, point2: pd.Series) -> float:
    """
    Calculates the euclidean distance between two points.
    """
    squares = []
    for point1_features, point2_features in zip(point1, point2):
        squares.append(np.square(point1_features - point2_features))

    return np.sqrt(np.sum(squares))