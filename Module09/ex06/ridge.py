import numpy as np
from my_linear_regression import MyLinearRegression as MLR
from l2_reg import l2
from vec_log_loss import vec_log_loss_ as v_log_loss_


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
    if (x.ndim == 1):
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

class MyRidge(MLR):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_
        if (isinstance(lambda_, int)):
            self.lambda_ = float(lambda_)
    
    def loss_(self, y, y_hat):
        """Computes the regularized loss of a logistic regression model from
        two non-empty numpy.ndarray, without any for l
        Args:
                y: has to be an numpy.ndarray, a vector of shape m * 1.
                y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
                The regularized loss as a float.
                None if y, y_hat, or theta is empty numpy.ndarray.
                None if y and y_hat do not share the same shapes.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(y_hat, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.lambda_, float)
        ):
            return None
        if y.size == 0 or y_hat.size == 0 or self.theta.size == 0:
            return None
        if(
            y.shape != y_hat.shape
            or y.shape[1] != 1
            or self.theta.shape[1] != 1
        ):
            return None
        m = y.shape[0]
        loss = v_log_loss_(y, y_hat) + self.lambda_ / (2 * m) * l2(self.theta)
        return loss

    def gradient_(self, y, x):  # où utilise-t-on l2 ?
        """Computes the regularized linear gradient of three non-empty
        numpy.ndarray, without any for-loop. The three arrays must have
        compatible shapes.
        Args:
                y: has to be a numpy.ndarray, a vector of shape m * 1.
                x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        Return:
                A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
                results of the formula for all j.
                None if y, x, or theta are empty numpy.ndarray.
                None if y, x or theta does not share compatibles shapes.
                None if y, x or theta or lambda_ is not of the expected type.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(x, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.lambda_, float)
        ):
            return None
        if y.size == 0 or x.size == 0 or self.theta.size == 0:
            return None
        if y.shape[0] != x.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        m = y.shape[0]
        x_prime = add_intercept(x)
        x_prime_T = np.transpose(x_prime)
        y_hat = np.matmul(x_prime, self.theta)
        theta_prime = self.theta
        theta_prime[0][0] = 0
        lambda_theta_prime = self.lambda_ * theta_prime
        grad = 1 / m * (np.matmul(x_prime_T, y_hat - y) + lambda_theta_prime)
        return grad

    # def fit_(self, ):  # où utilise-t-on l2 ?
