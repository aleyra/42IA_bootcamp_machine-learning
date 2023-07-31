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
    # mlr1 = sami(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-4, 
    #     max_iter = 1000, 
    #     x = X_training, 
    #     y = small_Y_training
    # )  # ok with alpha = 1.0e-7 and max_iter = 370000
    # print(f"theta = {mlr1.theta} alpha = {mlr1.alpha} max_iter = {mlr1.max_iter}")
    # mlr1 = MLR(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-7,
    #     max_iter = 370000
    # )  # pour aller plus vite pendant les tests
    # mlr1.fit_(X_training, small_Y_training)
    # small_Y_hat1 = mlr1.predict_(X_test)
    # loss1 = mlr1.loss_(small_Y_test, small_Y_hat1)
    # mse1 = mlr1.mse_(small_Y_test, small_Y_hat1)
    # print(f"when degree = 1 : mse = {mse1} and loss = {loss1}")
    # when degree = 1 : mse = 4.348879397721001 and loss = 2.1744396988605006

    # Model 2 : degree = 2
    X_training2 = apftm(X_training, 2)
    X_test2 = apftm(X_test, 2)
    theta_size = 1 + X_training2.shape[1]
    # mlr2 = sami(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-4, 
    #     max_iter = 1000, 
    #     x = X_training2, 
    #     y = small_Y_training
    # )  # ok with alpha = 1.0e-14? and max_iter = ? c'est trop long !!!!
    # print(f"theta = {mlr2.theta} alpha = {mlr2.alpha} max_iter = {mlr2.max_iter}")
    mlr2 = MLR(
        theta = np.ones(theta_size).reshape(-1,1),
        alpha = 1.0e-14,
        max_iter = 10000000
    )  # pour aller plus vite pendant les tests
    res = mlr2.fit_(X_training2, small_Y_training)
    print(res)
    # small_Y_hat2 = mlr2.predict_(X_test2)
    # loss2 = mlr2.loss_(small_Y_test, small_Y_hat2)
    # mse2 = mlr2.mse_(small_Y_test, small_Y_hat2)
    # print(f"when degree = 1 : mse = {mse2} and loss = {loss2}")

    # Model 3 : degree = 3
    # X_training3 = apftm(X_training, 3)
    # X_test3 = apftm(X_test, 3)
    # theta_size = 1 + X_training3.shape[1]
    # mlr3 = sami(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-4, 
    #     max_iter = 1000, 
    #     x = X_training3, 
    #     y = small_Y_training
    # )  # ok with alpha = ? and max_iter = ? c'est trop long !!!!
    # print(f"theta = {mlr3.theta} alpha = {mlr3.alpha} max_iter = {mlr3.max_iter}")
    # mlr3 = MLR(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-14,
    #     max_iter = 1000000
    # )  # pour aller plus vite pendant les tests
    # res = mlr3.fit_(X_training3, small_Y_training)
    # print(res)
    # small_Y_hat3 = mlr3.predict_(X_test3)
    # loss3 = mlr3.loss_(small_Y_test, small_Y_hat3)
    # mse3 = mlr3.mse_(small_Y_test, small_Y_hat3)
    # print(f"when degree = 1 : mse = {mse3} and loss = {loss3}")

    # Model 4 : degree = 4
    # X_training4 = apftm(X_training, 4)
    # X_test4 = apftm(X_test, 4)
    # theta_size = 1 + X_training4.shape[1]
    # mlr4 = sami(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-4, 
    #     max_iter = 1000, 
    #     x = X_training4, 
    #     y = small_Y_training
    # )  # ok with alpha = ? and max_iter = ? c'est trop long !!!!
    # print(f"theta = {mlr4.theta} alpha = {mlr4.alpha} max_iter = {mlr4.max_iter}")
    # mlr4 = MLR(
    #     theta = np.ones(theta_size).reshape(-1,1),
    #     alpha = 1.0e-14,
    #     max_iter = 1000000
    # )  # pour aller plus vite pendant les tests
    # res = mlr4.fit_(X_training4, small_Y_training)
    # print(res)
    # small_Y_hat4 = mlr4.predict_(X_test4)
    # loss4 = mlr4.loss_(small_Y_test, small_Y_hat4)
    # mse4 = mlr4.mse_(small_Y_test, small_Y_hat4)
    # print(f"when degree = 1 : mse = {mse4} and loss = {loss4}")
