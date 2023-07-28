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
    
    # make Ys smaller
    Y_training /= 100000
    Y_test /= 100000

    # Model 1 : degree = 1
    theta_size = 1 + X_training.shape[1]
    # mlr1 = sami(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-4, 
    #     max_iter = 1000, 
    #     x = X_training, 
    #     y = Y_training
    # )  # ok with alpha = 1.0e-7 and max_iter = 300000

    mlr1 = MLR(
        theta = np.ones(theta_size).reshape(-1,1),
        alpha = 1.0e-7, 
        max_iter = 400000, 
    )
    res = mlr1.fit_(
        x = X_training, 
        y = Y_training
    )
    print(f"alpha = {mlr1.alpha} max_iter = {mlr1.max_iter} res = {res}")
    # print(f"theta = {mlr1.theta} alpha = {mlr1.alpha} max_iter = {mlr1.max_iter}")
