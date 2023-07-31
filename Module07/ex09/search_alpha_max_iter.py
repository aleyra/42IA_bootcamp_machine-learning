import numpy as np
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features

def search_alpha(theta, alpha, max_iter, x, y):
    mlr = MLR(theta, alpha, max_iter)
    res = mlr.fit_(x, y)
    first_wall = max((max(y),min(y))) * 100
    # bool_t = True  # debug
    while (isinstance(res, str) or res >first_wall[0] or res < -first_wall[0]):
        # if bool_t == True:  # debug
        #     print("ds 1e while")
        #     bool_t = False
        alpha = alpha / 10
        mlr = MLR(theta, alpha, max_iter)
        # mlr.alpha = mlr.alpha / 10  # l'idée était de remplacer L14-15 par ça
        res = mlr.fit_(x, y)
        print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
    return alpha

def search_max_iter_new_theta(theta, alpha, x, y):
    mlr = MLR(theta, alpha, 1000)
    res = mlr.fit_(x, y)
    max_iter = 0
    adding = 100000
    sign = 1
    while (adding > 10) :
        i = 0
        if res > 0:
            # print("res positif")  # debug
            while res > 0.01 and i < 10:
                # if bool_t == True:  # debug
                #     print("ds 2e while")
                #     bool_t = False
                max_iter += sign * adding
                mlr = MLR(theta, alpha, max_iter)
                # mlr.max_iter = mlr.max_iter + sign * adding  # meme idée que pour alpha...
                res = mlr.fit_(x, y)
                print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
                i += 1
        elif res < 0:
            # print("res negatif")  # debug
            while res < -0.01 and i < 10:
                # if bool_t == True:  # debug
                #     print("ds 2e while")
                #     bool_t = False
                max_iter += sign * adding
                mlr = MLR(theta, alpha, max_iter)
                res = mlr.fit_(x, y)
                print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
                i += 1
        adding = int(adding / 10)
        sign *= -1
    return mlr

def search_alpha_max_iter(theta, alpha, max_iter, x, y):
    alpha = search_alpha(theta, alpha, max_iter, x, y)
    mlr = search_max_iter_new_theta(theta, alpha, x, y)
    return mlr

if __name__ == "__main__":
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
    
    theta = np.ones(4).reshape(-1,1)
    alpha = 1.0e-4
    max_iter = 1000

    x_ = add_polynomial_features(x, 3)
    # get a better alpha, max_iter and theta :) as setting mlr
    mlr = search_alpha_max_iter(theta, alpha, max_iter, x_, y)
    if not isinstance(mlr, MLR):
        print("error")
        exit()
    print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter}")
    


