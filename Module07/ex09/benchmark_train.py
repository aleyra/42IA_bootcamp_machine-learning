import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features
from data_spliter import data_spliter
from search_alpha_max_iter import search_alpha_max_iter as sami

if __name__ == "__main__":
    # from csv to np.ndarray
    data = pd.read_csv("../data/space_avocado.csv")
    Ytarget = np.array(data['target']).reshape(-1,1)
    X = np.array(data[['weight', 'prod_distance', 'time_delivery']])

    # split in training set and test set
    split = data_spliter(X, Ytarget, 0.8)
    X_training = split[0]
    Y_training = split[2]
    X_test = split[1]
    Y_test = split[3]

    