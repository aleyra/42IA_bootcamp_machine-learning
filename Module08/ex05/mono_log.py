import sys
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MLR
from data_spliter import data_spliter

def to_label_by_zipcode(x, zipcode):
    ret = x
    for i in range(ret.shape[0]):
        if ret[i][0] == zipcode:
            ret[i][0] = 1
        else:
            ret[i][0] = 0
    return ret
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Only 1 argument : -zipcode=x with x being 0, 1, 2 or 3")
        exit()
    if sys.argv[1][:9] != "-zipcode=" or len(sys.argv[1]) != 10 :
        print("Only 1 argument : -zipcode=x with x being 0, 1, 2 or 3")
        exit()
    zipcode = ord(sys.argv[1][-1:]) - ord('0')
    if zipcode < 0 or zipcode > 3:
        print("Only 1 argument : -zipcode=x with x being 0, 1, 2 or 3")
        exit()
    print(f"zipcode = {zipcode}")  #

    data = pd.read_csv("../data/solar_system_census_planets.csv")
    origin = np.array(data['Origin']).reshape(-1,1)
    lst = []
    for i in range(data.shape[0]):
        lst.append(i)
    x = np.array(lst).reshape(-1,1)

    # split dataset
    split = data_spliter(x, origin, 0.8)
    X_training = split[0]
    X_test = split[1]
    Y_training = split[2]
    Y_test = split[3]

    # new np.array to label each citizen according to your new selection criterion
    labeled_Y_training = to_label_by_zipcode(Y_training, zipcode)
    labeled_Y_test = to_label_by_zipcode(Y_test, zipcode)

    # Train a logistic model
    theta = np.array([1, 1]).reshape(-1, 1)
    mlr = MLR(
        theta = theta, 
        alpha = 0.001,
        max_iter = 50000
    )
    res = mlr.fit_(X_training, labeled_Y_training)
    print(f"max_iter = {mlr.max_iter} et res = {res}")
    # 

