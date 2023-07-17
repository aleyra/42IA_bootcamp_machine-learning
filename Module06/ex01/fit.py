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
    if x.shape != y.shape or x.shape[1] != 1 or theta.shape != (2, 1):
        return None
    x_prime = add_intercept(x)
    x_primet_over_m = 1 / x.shape[0] * np.transpose(x_prime)
    prodmat = np.matmul(x_prime, theta)
    t = prodmat - y
    gradient = np.matmul(x_primet_over_m, t)
    return gradient


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            function should not raise any Exceptions.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (x.size == 0 or theta.size == 0):
        return None
    if (x.ndim != 2 or x.shape[1] != 1 or theta.shape != (2, 1)):
        return None
    x = add_intercept(x)
    y_hat = np.matmul(x, theta)
    return y_hat


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during
            the gradient descent
    Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(alpha, float)
        or not isinstance(max_iter, int)
    ):
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if alpha > 1 or alpha < 0 or max_iter < 1:
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape != y.shape or x.shape[1] != 1 or theta.shape != (2, 1):
        return None

    new_theta = theta
    for i in range(max_iter):
        new_theta = new_theta - alpha * simple_gradient(x, y, new_theta)
    return new_theta
    

if __name__ == "__main__":
    lstx = [
        [12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]
    ]
    x = np.array(lstx)

    lsty = [
        [37.4013816],[36.1473236], [45.7655287], [46.6793434], [59.5585554]
    ]
    y = np.array(lsty)
    theta= np.array([1, 1]).reshape((-1, 1))
    
    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    # Output:
    # array([[1.40709365],
    #         [1.1150909 ]])
    
    # Example 1:
    print(predict_(x, theta1))  # function from Module05 ex02 almost -_-
    # Output:
    # array([[15.3408728 ],
    #         [25.38243697],
    #         [36.59126492],
    #         [55.95130097],
    #         [65.53471499]])