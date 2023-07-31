import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_featured_to_matrix as apftm
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
    small_Y_training = Y_training / 100000
    small_Y_test = Y_test / 100000

    # Model 1 : degree = 1
    theta_size = 1 + X_training.shape[1]
    mlr1 = sami(
        theta = np.ones(theta_size).reshape(-1,1),
        alpha = 1.0e-4, 
        max_iter = 1000, 
        x = X_training, 
        y = small_Y_training
    )  # ok with alpha = 1.0e-7 and max_iter = 370000
    print(f"theta = {mlr1.theta} alpha = {mlr1.alpha} max_iter = {mlr1.max_iter}")
    mlr1 = MLR(
        theta = np.ones(theta_size).reshape(-1,1),
        alpha = 1.0e-7,
        max_iter = 370000
    )  # pour aller plus vite pendant les tests
    Y_hat1 = mlr1.predict_(X_test)
    loss1 = mlr1.loss_(Y_test, Y_hat1)
    mse1 = mlr1.mse_(Y_test, Y_hat1)
    print(f"when degree = 1 : mse = {mse1} and loss = {loss1}")
