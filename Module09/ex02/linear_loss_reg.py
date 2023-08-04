import numpy as np
from l2_reg import l2


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two
    non-empty numpy.array, without any for loop.
    Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
    Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(lambda_, float)
    ):
        print("y or h_hat or theta is not a np.ndarray or lambda is not a float")
        return None
    if y.size == 0 or y_hat.size == 0 or theta.size == 0:
        print("y or y_hat or theta is empty")
        return None
    if y.shape != y_hat.shape or y.shape[1] != 1 or theta.shape[1] != 1:
        print("y or y_hat or theta are not vectors or y and y_hat doesn't have the same shape")
        return None
    m = y.shape[0]
    diff = y_hat - y
    diff2 = np.dot(np.transpose(diff), diff)
    loss = 1 / (2 * m) * (diff2 + lambda_ * l2(theta))
    return loss[0][0]


if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(reg_loss_(y, y_hat, theta, .5))
    # Output:
    0.8503571428571429

    # Example :
    print(reg_loss_(y, y_hat, theta, .05))
    # Output:
    0.5511071428571429

    # Example :
    print(reg_loss_(y, y_hat, theta, .9))
    # Output:
    1.116357142857143
