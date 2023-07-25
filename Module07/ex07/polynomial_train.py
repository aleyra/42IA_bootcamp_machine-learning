import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


if __name__ == "__main__":
    # 1st part
    x = np.arange(1,11).reshape(-1,1)
    y = np.array([[ 1.39270298],
        [ 3.88237651],
        [ 4.37726357],
        [ 4.63389049],
        [ 7.79814439],
        [ 6.41717461],
        [ 8.63429886],
        [ 8.19939795],
        [10.37567392],
        [10.68238222]])
    
    # plt.scatter(x,y)
    # plt.show()

    f_x = open('x.csv', 'w')
    writer_x = csv.writer(f_x)
    for i in range(x.shape[0]):
        writer_x.writerow(x[i])
    f_x.close()

    f_y = open('y.csv', 'w')
    writer_y = csv.writer(f_y)
    for i in range(y.shape[0]):
        writer_y.writerow(y[i])
    f_y.close()

    # 2nd part
    # Build the model:
    x_ = add_polynomial_features(x, 3)
    my_lr = MyLR(np.ones(4).reshape(-1,1), alpha= 1.0e-6)
    # without alpha we've got an error
    my_lr.fit_(x_, y)
    # print(my_lr.theta)

    

    # Plot:
    ## To get a smooth curve, we need a lot of data points
    continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
    x_ = add_polynomial_features(continuous_x, 3)
    y_hat = my_lr.predict_(x_)

    f_continuous_x = open('continuous_x.csv', 'w')
    writer_continuous_x = csv.writer(f_continuous_x)
    for i in range(continuous_x.shape[0]):
        writer_continuous_x.writerow(continuous_x[i])
    f_continuous_x.close()

    f_y_hat = open('y_hat.csv', 'w')
    writer_y_hat = csv.writer(f_y_hat)
    for i in range(y_hat.shape[0]):
        writer_y_hat.writerow(y_hat[i])
    f_y_hat.close()
    
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

    my_lr4 = MyLR(theta=theta4, alpha=1.0e-4)
    my_lr5 = MyLR(theta=theta5, alpha=1.0e-4)
    my_lr6 = MyLR(theta=theta6, alpha=1.0e-4)    
