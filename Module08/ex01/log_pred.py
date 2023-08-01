import numpy as np
from sigmoid import sigmoid_
from tools import add_intercept

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray
    Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
            theta: has to be an numpy.ndarray, 
            a vector of dimension (n + 1) * 1.
    Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
    Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("x or theta not a np.ndarray")
        return None
    if theta.size == 0 or x.size == 0:
        print("x or theta is empty")
        return None
    if theta.shape[0] != x.shape[1] + 1:
        print("theta.shape[0] + 1 != x.shape[1]")
        return None
    x_prime = add_intercept(x)
    y_hat = sigmoid_(np.matmul(x_prime, theta))
    return y_hat

if __name__ == "__main__":
    # Example 1
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    y_hat = logistic_predict_(x, theta)
    print(y_hat)
    # Output:
    # array([[0.98201379]])
    
    # Example 2
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat = logistic_predict_(x2, theta2)
    print(y_hat)
    # Output:
    # array([[0.98201379],
    #     [0.99624161],
    #     [0.97340301],
    #     [0.99875204],
    #     [0.90720705]])
    
    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat = logistic_predict_(x3, theta3)
    print(y_hat)
    # Output:
    # array([[0.03916572],
    #     [0.00045262],
    #     [0.2890505 ]])