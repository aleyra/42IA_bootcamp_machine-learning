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


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
            x: has to be an numpy.array, a vector of dimensions m * n.
            theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1
    Return:
            y_hat as a numpy.array, a vector of dimensions m * 1.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
            None if x or theta is not of expected type.
    Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if theta.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0]:
        return None
    y_hat = np.ndarray((x.shape[0], 1))
    x = add_intercept(x)
    y_hat = np.matmul(x, theta)
    return y_hat


if __name__ == "__main__":
    x = np.arange(1, 13).reshape((4, -1))

    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    p = predict_(x, theta1)
    print(f"exemple 1\n{p}")
    # Ouput:
    # array([[5.], [5.], [5.], [5.]])
    # Do you understand why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    p = predict_(x, theta2)
    print(f"exemple 2\n{p}")
    # Output:
    # array([[ 1.], [ 4.], [ 7.], [10.]])
    # Do you understand why y_hat == x[:,0] here?

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    p = predict_(x, theta3)
    print(f"exemple 3\n{p}")
    # Output:
    # array([[ 9.64], [24.28], [38.92], [53.56]])

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    p = predict_(x, theta4)
    print(f"exemple 4\n{p}")
    # Output:
    # array([[12.5], [32. ], [51.5], [71. ]])
