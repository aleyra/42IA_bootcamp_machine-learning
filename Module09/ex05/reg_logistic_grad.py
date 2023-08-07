import numpy as np
from tools import add_intercept
from sigmoid import sigmoid_


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty
    numpy.ndarray, with two for-loops. The three array
    Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape n + 1 * 1.
            lambda_: has to be a float.
    Returns:
            A numpy.ndarray, a vector of shape n * 1, containing the results
            of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
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
    if y.shape[0] != x.shape[0] or x.shape[1] +1 != theta.shape[0]:
        return None

    m = y.shape[0]
    x_prime = add_intercept(x)
    y_hat = sigmoid_(np.matmul(x_prime, theta))
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


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty
    numpy.ndarray, without any for-loop. The three arr
    Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of shape m * n.
            theta: has to be a numpy.ndarray, a vector of shape n + 1 * 1.
            lambda_: has to be a float.
    Returns:
            A numpy.ndarray, a vector of shape n * 1, containing the results
            of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
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
    y_hat = sigmoid_(np.matmul(x_prime, theta))
    theta_prime = theta
    theta_prime[0][0] = 0
    grad = 1 / m * (np.matmul(x_prime_T, y_hat - y) + lambda_ * theta_prime)
    return grad


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1.1:
    print(reg_logistic_grad(y, x, theta, 1))
    # Output:
    # array([[-0.55711039],
    #     [-1.40334809],
    #     [-1.91756886],
    #     [-2.56737958],
    #     [-3.03924017]])

    # Example 1.2:
    print(vec_reg_logistic_grad(y, x, theta, 1))
    # Output:
    # array([[-0.55711039],
    #     [-1.40334809],
    #     [-1.91756886],
    #     [-2.56737958],
    #     [-3.03924017]])

    # Example 2.1:
    print(reg_logistic_grad(y, x, theta, 0.5))
    # Output:
    # array([[-0.55711039],
    #     [-1.15334809],
    #     [-1.96756886],
    #     [-2.33404624],
    #     [-3.15590684]])

    # Example 2.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    # Output:
    # array([[-0.55711039],
    #     [-1.15334809],
    #     [-1.96756886],
    #     [-2.33404624],
    #     [-3.15590684]])

    # Example 3.1:
    print(reg_logistic_grad(y, x, theta, 0.0))
    # Output:
    # array([[-0.55711039],
    #     [-0.90334809],
    #     [-2.01756886],
    #     [-2.10071291],
    #     [-3.27257351]])

    # Example 3.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.0))
    # Output:
    # array([[-0.55711039],
    #     [-0.90334809],
    #     [-2.01756886],
    #     [-2.10071291],
    #     [-3.27257351]])
