import numpy as np


class MyLinearRegression():
    """
    Description:
            An improves version of MyLinearRegression class made in
            Module06/ex02
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if (isinstance(theta, list)):
            self.theta = np.array(theta)
        else:
            self.theta = theta

    def add_intercept(self, x):
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
    
    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without
        any for-loop.
        The three arrays must have the compatible dimensions.
        Args:
                x: has to be an numpy.array, a matrix of dimension m * n.
                y: has to be an numpy.array, a vector of dimension m * 1.
                theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
                The gradient as a numpy.array, a vector of dimensions n * 1,
                containg the result of the formula for all j.
                None if x, y, or theta are empty numpy.array.
                None if x, y and theta do not have compatible dimensions.
                None if x, y or theta is not of expected type.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
        ):
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        x_prime = self.add_intercept(x)
        x_prime_T = np.transpose(x_prime)
        prodmat = np.matmul(x_prime, self.theta)
        gradient = np.matmul(x_prime_T, prodmat - y) / x.shape[0]
        return gradient
    
    def fit_(self, x, y):
        """
        Description:
                Fits the model to the training dataset contained in x and y.
        Args:
                x: has to be a numpy.array, a matrix of dimension m * n:
                (number of training examples, number of features).
                y: has to be a numpy.array, a vector of dimension m * 1:
                (number of training examples, 1).
                theta: has to be a numpy.array, a vector
                of dimension (n + 1) * 1: (number of features + 1, 1).
                alpha: has to be a float, the learning rate
                max_iter: has to be an int, the number of iterations done
                during the gradient descent
        Return:
                new_theta: numpy.array, a vector of dimension
                (number of features + 1, 1).
                None if there is a matching dimension problem.
                None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            return None
        if x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2:
            return None
        if self.alpha > 1 or self.alpha < 0 or self.max_iter < 1:
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        
        for i in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient(x, y)
        return self.theta

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
                x: has to be an numpy.array, a vector of dimensions m * n.
                theta: has to be an numpy.array, a vector of
                dimensions (n + 1) * 1
        Return:
                y_hat as a numpy.array, a vector of dimensions m * 1.
                None if x or theta are empty numpy.array.
                None if x or theta dimensions are not appropriate.
                None if x or theta is not of expected type.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
        ):
            return None
        if x.size == 0 or self.theta.size == 0:
            return None
        if self.theta.shape[1] != 1 or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        y_hat = np.ndarray((x.shape[0], 1))
        x = self.add_intercept(x)
        y_hat = np.matmul(x, self.theta)
        return y_hat
    
    def loss_(self, y, y_hat):
        """Computes the mean squared error of two non-empty numpy.array, without
        any for loop.
        The two arrays must have the same dimensions.
        Args:
                y: has to be an numpy.array, a vector.
                y_hat: has to be an numpy.array, a vector.
        Return:
                The mean squared error of the two vectors as a float.
                None if y or y_hat are empty numpy.array.
                None if y and y_hat does not share the same dimensions.
                None if y or y_hat is not of expected type.
        Raises:
                This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        diff = y_hat - y
        res = 0.0
        for i in range(diff.shape[0]):
            res += diff[i][0] * diff[i][0]
        res /= 2 * diff.shape[0]
        return res