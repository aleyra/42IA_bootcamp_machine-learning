import numpy as np
from tools import add_intercept


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty
    numpy.ndarray, with two for-loop. The three arrays must have compatible
    shapes.
    Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
    Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
            results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
    Raises:
            This function should not raise any Exception.
    """
    if isinstance(lambda_, int):
        lambda_ = float(lambda_)
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(lambda_, float)
    ):
        print("pb of type")
        return None
    if y.size == 0 or x.size == 0 or theta.size == 0:
        print("smthg empty")
        return None
    if y.shape[0] != x.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        print("pb of shape")
        return None
    m = y.shape[0]

    x_prime = add_intercept(x)
    y_hat = np.matmul(x_prime, theta)
    grad = np.ones(theta.shape)
    sum = 0
    for i in range(x.shape[0]):
        sum += y_hat[i][0] - y[i][0]
    grad[0][0] = 1 / m * sum

    for j in range(1, grad.shape[0]):
        sum = 0
        for i in range(x.shape[0]):
            sum += (y_hat[i][0] - y[i][0]) * x_prime[i][j]
        grad[j][0] = 1 / m * (sum + lambda_ * theta[j][0])
    return grad


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty
    numpy.ndarray, without any for-loop. The three arrays must have compatible
    shapes.
    Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
    Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
            results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
    Raises:
            This function should not raise any Exception.
    """
    if isinstance(lambda_, int):
        lambda_ = float(lambda_)
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(lambda_, float)
    ):
        return None
    if y.size == 0 or x.size == 0 or theta.size == 0:
        return None
    if y.shape[0] != x.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    m = y.shape[0]
    x_prime = add_intercept(x)
    x_prime_T = np.transpose(x_prime)
    y_hat = np.matmul(x_prime, theta)
    theta_prime = theta
    theta_prime[0][0] = 0
    grad = 1 / m * (np.matmul(x_prime_T, y_hat - y) + lambda_ * theta_prime)
    return grad


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    # Example 1.1:
    # print(reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])

    # Example 1.2:
    # print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])

    # Example 2.1:
    # print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])

    # Example 2.2:
    # print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])

    # Example 3.1:
    # print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])

    # Example 3.2:
    # print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])
