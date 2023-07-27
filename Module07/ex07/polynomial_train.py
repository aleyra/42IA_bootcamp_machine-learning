import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


if __name__ == "__main__":
    # 1st part
    # x = np.arange(1,11).reshape(-1,1)
    # y = np.array([[ 1.39270298],
    #     [ 3.88237651],
    #     [ 4.37726357],
    #     [ 4.63389049],
    #     [ 7.79814439],
    #     [ 6.41717461],
    #     [ 8.63429886],
    #     [ 8.19939795],
    #     [10.37567392],
    #     [10.68238222]])
    
    # plt.scatter(x,y)
    # plt.show()

    # 2nd part
    # Build the model:
    # x_ = add_polynomial_features(x, 3)
    # my_lr = MyLR(np.ones(4).reshape(-1,1), alpha= 9.0e-6, max_iter=6800)
    # without alpha we've got an error and without max_iter gradient was too big
    # my_lr.fit_(x_, y)
    # print(my_lr.theta)

    # Plot:
    ## To get a smooth curve, we need a lot of data points
    # continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
    # x_ = add_polynomial_features(continuous_x, 3)
    # y_hat = my_lr.predict_(x_)

    # print(y_hat[600])
    
    # plt.scatter(x,y)
    # plt.plot(continuous_x, y_hat, color='orange')
    # plt.show()

    
    # starting points ?
    theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
    theta5 = np.array(
        [[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
    theta6 = np.array(
        [[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]
    ).reshape(-1,1)

    data = pd.read_csv("../data/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)

    x_4 = add_polynomial_features(Xpill, theta4.shape[0] - 1)
    x_5 = add_polynomial_features(Xpill, theta5.shape[0] - 1)
    x_6 = add_polynomial_features(Xpill, theta6.shape[0] - 1)

    my_lr4 = MyLR(theta=theta4, alpha=2.2e-6, max_iter=10700)
    my_lr5 = MyLR(theta=theta5, alpha=7.2e-9, max_iter=100000)
    my_lr6 = MyLR(theta=theta6, alpha=1.0e-9, max_iter=22400)

    my_lr4.fit_(x_4, Yscore)
    my_lr5.fit_(x_5, Yscore)
    my_lr6.fit_(x_6, Yscore)

    min = min(Xpill)
    max = max(Xpill) + 0.01
    continuous_x = np.arange(min, max, 0.01).reshape(-1,1)

    x_4 = add_polynomial_features(continuous_x, theta4.shape[0] - 1)
    x_5 = add_polynomial_features(continuous_x, theta5.shape[0] - 1)
    x_6 = add_polynomial_features(continuous_x, theta6.shape[0] - 1)

    y_hat4 = my_lr4.predict_(x_4)
    y_hat5 = my_lr5.predict_(x_5)
    y_hat6 = my_lr6.predict_(x_6)

    plt.scatter(Xpill,Yscore)
    plt.plot(continuous_x, y_hat4, color='orange')
    plt.plot(continuous_x, y_hat5, color='red')
    plt.plot(continuous_x, y_hat6, color='purple')
    plt.show()



