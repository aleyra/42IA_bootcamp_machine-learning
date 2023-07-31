import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the
    power given in argument.
    Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            power: has to be an int, the power up to which the components of
            vector x are going to be raised.
    Return:
            The matrix of polynomial features as a numpy.array,
            of dimension m * n,
            containing the polynomial feature values for all training examples.
            None if x is an empty numpy.array.
            None if x or power is not of expected type.
    Raises:
            This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(power, int)):
        return None
    if x.size == 0 or power <= 0:
        return None
    res = x
    for i in range(2, power + 1):
        x_power_i = np.power(x, i)
        res = np.append(res, x_power_i, axis = 1)
    return res

def add_polynomial_featured_to_matrix(x, power):
    """
    use add_polynomial_featured to every column of a matrix
    """
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    if x.size == 0 or power <= 0:
        return None
    for i in range(x.shape[1]):
        col = x[:, i].reshape((-1, 1))
        if i == 0:
            res = add_polynomial_features(col, power)
        else:
            inter_res = add_polynomial_features(col, power)
            res = np.concatenate((res, inter_res), axis = 1)
    return res

if __name__ == "__main__":
    x = np.array([0,1,2,3,4,5,6,7,8,9]).reshape((-1,2))
    
    res = add_polynomial_featured_to_matrix(x, 2)
    print(res)