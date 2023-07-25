import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR


if __name__ == "__main__":
    data = pd.read_csv("../../data/spacecraft_data.csv")
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    # print(f"X = {X}")
    my_lreg = MyLR(
        theta = [1.0, 1.0, 1.0, 1.0], alpha = 9e-5, max_iter = 600000
    ) # alpha = 1e-4 dans l'énoncé de l'exercice

    # Example 0:
    y_hat = my_lreg.predict_(X)
    m = my_lreg.mse_(Y, y_hat)
    print(f"exemple 0\n{m}")
    # Output:
    # 144044.877...

    # my_lreg.gradient(X,Y)

    # Example 1:
    my_lreg.fit_(X,Y)
    print(f"exemple 1\n{my_lreg.theta}")
    # Output:
    # array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

    # Example 2:
    m = my_lreg.mse_(X,Y)
    print(f"exemple 2\n{m}")
    # Output:
    # 586.896999...