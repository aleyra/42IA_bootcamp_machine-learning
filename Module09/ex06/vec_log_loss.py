import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
    Returns:
            The logistic loss value as a float.
            None on any error.
    Raises:
            This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if not isinstance(eps, float) or eps <= 0 or eps >= 1:
        return None
    if y.shape != y_hat.shape:
        return None
    ones = np.ones(y.size).reshape(y.shape)
    diff = ones - y_hat
    for i in range(y_hat.shape[0]):
        if y_hat[i][0] == 0:
            y_hat[i][0] = eps
        if diff[i][0] == 0:
            diff[i][0] = eps
    log1 = np.log(y_hat)
    log2 = np.log(diff)
    m = y.shape[0]
    dot1 = np.matmul(np.transpose(y), log1)
    dot2 = np.matmul(np.transpose(ones - y), log2)
    res = 1 / -m * (dot1 + dot2)
    return res[0][0]
