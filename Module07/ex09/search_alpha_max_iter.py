import numpy as np
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features

def search_alpha(theta, alpha, max_iter, x, y):
    mlr = MLR(theta, alpha, max_iter)
    res = mlr.fit_(x, y)
    first_wall = max((max(y),min(y))) * 10
    # bool_t = True  # debug
    while (isinstance(res, str) or res >first_wall[0] or res < -first_wall[0]):
        # if bool_t == True:  # debug
        #     print("ds 1e while")
        #     bool_t = False
        alpha = alpha / 10
        mlr = MLR(theta, alpha, max_iter)
        # mlr.alpha = mlr.alpha / 10  # l'idée était de remplacer L15-16 par ça
        res = mlr.fit_(x, y)
        print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
    return alpha

def search_max_iter_new_theta(theta, alpha, max_iter, x, y):
    mlr = MLR(theta, alpha, max_iter)
    res = mlr.fit_(x, y)
    # bool_t = True  # debug
    if res > 0:
        while res > 0.01 :
            # if bool_t == True:  # debug
            #     print("ds 2e while")
            #     bool_t = False
            max_iter += 1000
            mlr = MLR(theta, alpha, max_iter)
            # mlr.max_iter = mlr.max_iter + 1000  # meme idée que pour alpha...
            res = mlr.fit_(x, y)
            print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
    elif res < 0:
        while res < -0.01 :
            # if bool_t == True:  # debug
            #     print("ds 2e while")
            #     bool_t = False
            max_iter += 1000
            mlr = MLR(theta, alpha, max_iter)
            res = mlr.fit_(x, y)
            print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res}")
    bool_t = True  # debug
    i = 0
    while (res > 0.01 or res < -0.01) and i != 10:
        # if bool_t == True:  # debug
        #     print("ds 3e while")
        #     bool_t = False
        max_iter -= 100
        mlr = MLR(theta, alpha, max_iter)
        res = mlr.fit_(x, y)
        i += 1
        print(f"alpha = {mlr.alpha} max_iter = {mlr.max_iter} res = {res} i = {i}")
    return (max_iter, mlr.theta)

def search_alpha_max_iter(theta, alpha, max_iter, x, y):
    alpha = search_alpha(theta, alpha, max_iter, x, y)
    max_iter_theta = search_max_iter_new_theta(theta, alpha, max_iter, x, y)
    mlr = MLR(max_iter_theta[1], alpha, max_iter[0])
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
    


