import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every
    power in the range of 1 up to the power give
    Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            power: has to be an int, the power up to which the columns of
            matrix x are going to be raised.
    Returns:
            The matrix of polynomial features as a numpy.ndarray, of shape
            m * (np), containg the polynomial feature va
            None if x is an empty numpy.ndarray.
    Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    if x.size == 0 or x.ndim != 2 or power <= 0:
        return None
    ret = x
    for i in range(2, power + 1):
        x_power_i = np.power(x[:, 0], i).reshape(-1, 1)
        for j in range(1, x.shape[1]):
            xj_power_i = np.power(x[:, j], i).reshape(-1, 1)
            x_power_i = np.append(x_power_i, xj_power_i, axis=1)
        ret = np.concatenate((ret, x_power_i), axis=1)
    return ret


if __name__ == "__main__":
    x = np.arange(1, 11).reshape(5, 2)
    print(x)

    # Example 1:
    bouh = add_polynomial_features(x, 3)
    print(bouh)

    # Example 2:
    bouh = add_polynomial_features(x, 4)
    print(bouh)
