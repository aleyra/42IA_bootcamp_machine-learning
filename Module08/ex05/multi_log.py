import sys
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MLR
from data_spliter import data_spliter


if __name__ == "__main__":
    data = pd.read_csv("../data/solar_system_census.csv")
    X = np.array(data[['weight', 'height']])
    Y = np.array(data['bone_density']).reshape(-1,1)
    
    # split data
    split = data_spliter(X, Y, 0.8)
    X_training = split[0]
    X_test = split[1]
    Y_training = split[2]
    Y_test = split[3]

    # train a logictic model
    theta = np.ones(X.shape[1] + 1).reshape(-1, 1)
    mlr = MLR(
        theta = theta,
        alpha = 0.001,
        max_iter = 1000
    )
    res = mlr.fit_(X_training, Y_training)
    print(f"max_iter = {mlr.max_iter} et res = {res} theta = {mlr.theta}")