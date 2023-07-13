import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    """
    Description:
            Calculate the MSE between the predicted output and the real
            output.
    Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
    Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != y_hat.ndim:
        return None
    res = 0.0
    # print(y[1])
    for i in range(y.shape[0]):
        res += (y_hat[i] - y[i]) * (y_hat[i] - y[i])
    res /= y.shape[0]
    return res


def rmse_(y, y_hat):
    """
    Description:
            Calculate the RMSE between the predicted output and the real
            output.
    Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
            rmse: has to be a float.
            None if there is a matching dimension problem.
    Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != y_hat.ndim:
        return None
    res = 0.0
    for i in range(y.shape[0]):
        res += (y_hat[i] - y[i]) * (y_hat[i] - y[i])
    res /= y.shape[0]
    return sqrt(res)


def mae_(y, y_hat):
    """
    Description:
            Calculate the MAE between the predicted output and the real
            output.
    Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
            mae: has to be a float.
            None if there is a matching dimension problem.
    Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != y_hat.ndim:
        return None
    res = 0.0
    for i in range(y.shape[0]):
        res += abs(y_hat[i] - y[i])
    res /= y.shape[0]
    return res


def r2score_(y, y_hat):
    """
    Description:
            Calculate the R2score between the predicted output and the
            output.
    Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
            r2score: has to be a float.
            None if there is a matching dimension problem.
    Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != y_hat.ndim:
        return None
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != y_hat.ndim:
        return None
    dividend = 0.0
    divisor = 0.0
    mean = 0.0
    for i in range(y.shape[0]):
        mean += y[i]
    mean /= y.shape[0]
    for i in range(y.shape[0]):
        dividend += (y_hat[i] - y[i]) * (y_hat[i] - y[i])
        divisor += (y_hat[i] - mean) * (y_hat[i] - mean)
    res = 1 - dividend / divisor
    return res


if __name__ == "__main__":
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    ## your implementation
    print(mse_(x, y))
    ## Output:
    # 4.285714285714286
    ## sklearn implementation
    print(mean_squared_error(x, y))
    ## Output:
    # 4.285714285714286

    # Root mean squared error
    ## your implementation
    print(rmse_(x, y))
    ## Output:
    # 2.0701966780270626
    ## sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    ## Output:
    # 2.0701966780270626

    # Mean absolute error
    ## your implementation
    print(mae_(x, y))
    # Output:
    # 1.7142857142857142
    ## sklearn implementation
    print(mean_absolute_error(x, y))
    # Output:
    # 1.7142857142857142

    # R2-score
    ## your implementation
    print(r2score_(x, y))
    ## Output:
    # 0.9681721733858745
    ## sklearn implementation
    print(r2_score(x, y))
    ## Output:
    # 0.9681721733858745
