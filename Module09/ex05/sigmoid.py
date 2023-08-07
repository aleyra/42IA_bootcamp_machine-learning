import numpy as np
from math import exp


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
            x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
            None if x is an empty numpy.ndarray.
    Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.ndim == 2 and x.shape[1] != 1:
        return None
    res = np.ones(x.size).reshape(x.shape)
    for i in range(x.shape[0]):
        res[i][0] = 1 / (1 + exp(-x[i][0]))
    return res
