import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a
    for-loop.
    Args:
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(theta, np.ndarray)
        or theta.size == 0
        or theta.shape[1] != 1
    ):
        return None
    l2 = 0
    theta_prime = theta
    theta_prime[0][0] = 0
    for i in range(theta_prime.shape[0]):
        l2 += theta_prime[i][0] * theta_prime[i][0]
    return l2


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any
    for-loop.
    Args:
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(theta, np.ndarray)
        or theta.size == 0
        or theta.shape[1] != 1
    ):
        return None
    theta_prime = theta
    theta_prime[0][0] = 0
    l2 = np.dot(np.transpose(theta_prime), theta_prime)
    return l2[0][0]


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    print(iterative_l2(x))
    # Output:
    911.0

    # Example 2:
    print(l2(x))
    # Output:
    911.0
    y = np.array([3, 0.5, -6]).reshape((-1, 1))

    # Example 3:
    print(iterative_l2(y))
    # Output:
    36.25

    # Example 4:
    print(l2(y))
    # Output:
    36.25
