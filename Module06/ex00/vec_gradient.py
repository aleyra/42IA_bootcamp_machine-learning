import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if x.ndim == 1:
        res = []
        nb_line = x.shape[0]
        for i in range(nb_line):
            res.append(1)
            res.append(x[i])
        res = np.array(res)
        res = res.reshape(nb_line, 2)
        return res

    col1 = list()
    for i in range(x.shape[0]):
        col1.append(1)
    res = np.insert(x, 0, col1, axis=1)
    return res


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any
    for loop.
    The three arrays must have compatible shapes.
    Args:
            x: has to be a numpy.array, a matrix of shape m * 1.
            y: has to be a numpy.array, a vector of shape m * 1.
            theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
            None if x, y, or theta is an empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
    ):
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape != y.shape or theta.shape != (2, 1):
        return None
    # print(x.shape)
    x_prime = add_intercept(x)
    x_primet_over_m = 1 / x.shape[0] * np.transpose(x_prime)
    prodmat = np.matmul(x_prime, theta)
    t = prodmat - y
    gradient = np.matmul(x_primet_over_m, t)
    return gradient


if __name__ == "__main__":
    lstx = [12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]
    x = np.array(lstx).reshape((-1, 1))
    lsty = [37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]
    y = np.array(lsty).reshape((-1, 1))

    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1))
    # Output:
    # array([[-19.0342...], [-586.6687...]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2))
    # Output:
    # array([[-57.8682...], [-2230.1229...]])
