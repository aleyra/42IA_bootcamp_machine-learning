import numpy as np
from log_pred import logistic_predict_


def log_to_mat(x):
    """
    apply log to all elem of
    """

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
    log1 = np.log(y_hat)
    log2 = np.log(ones - y_hat)
    m = y.shape[0]
    dot1 = np.matmul(np.transpose(y), log1)
    dot2 = np.matmul(np.transpose(ones - y), log2)
    res = 1 / -m * (dot1 + dot2)
    return res[0][0]

if __name__ == "__main__":
    # Example 1:
    # y1 = np.array([1]).reshape((-1, 1))
    # x1 = np.array([4]).reshape((-1, 1))
    # theta1 = np.array([[2], [0.5]])
    # y_hat1 = logistic_predict_(x1, theta1)
    # loss = vec_log_loss_(y1, y_hat1)
    # print(loss)
    # Output:
    # 0.018149927917808714

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    loss = vec_log_loss_(y2, y_hat2)
    print(loss)
    # Output:
    # 2.4825011602472347

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    loss = vec_log_loss_(y3, y_hat3)
    print(loss)
    # Output:
    # 2.993853310859968