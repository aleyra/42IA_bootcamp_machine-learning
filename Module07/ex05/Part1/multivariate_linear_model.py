import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR


if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    myLR_age = MyLR(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(X[:,0].reshape(-1,1), Y)

    y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
    print(myLR_age.mse_(y_pred,Y))
    #Output
    # 55736.86719...