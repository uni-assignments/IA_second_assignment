
def read_data(path):
    return pd.read_csv(path, sep=";", header = None, index_col = False)

def divide_features_and_label(train_data):
    return train_data.iloc[:, :-1], train_data.iloc[:, -1].tolist()

def euclidean_distance(new_sample, classified_element):

    squares = []
    for new_sample_feature, classified_feature in zip(new_sample, classified_element):
        squares.append(np.square(new_sample_feature - classified_feature))

    return np.sqrt(np.sum(squares))


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