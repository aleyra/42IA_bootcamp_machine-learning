import numpy as np


class MyLinearRegression():
    """
    Description:
            My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

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
    
    def simple_gradient(self, x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without
        any for loop.
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
        x_prime = self.add_intercept(x)
        x_primet_over_m = 1 / x.shape[0] * np.transpose(x_prime)
        prodmat = np.matmul(x_prime, theta)
        t = prodmat - y
        gradient = np.matmul(x_primet_over_m, t)
        return gradient
    
    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
                x: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
                y: has to be a numpy.ndarray, a vector of dimension m * 1:
                (number of training examples, 1).
        Returns:
                new_theta: numpy.ndarray, a vector of dimension 2 * 1.
                None if there is a matching dimension problem.
        Raises:
                This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            return None
        if x.ndim != 2 or y.ndim != 2 or self.thetas.ndim != 2:
            return None
        if self.alpha > 1 or self.alpha < 0 or self.max_iter < 1:
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if x.shape != y.shape or x.shape[1] != 1 or self.thetas.shape != (2, 1):
            return None

        new_theta = self.thetas
        for i in range(self.max_iter):
            new_theta = new_theta - self.alpha * self.simple_gradient(
                x, y, new_theta
            )
        self.thetas = new_theta
        return new_theta

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            function should not raise any Exceptions.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
        ):
            return None
        if (x.size == 0 or self.thetas.size == 0):
            return None
        if (x.ndim != 2 or x.shape[1] != 1 or self.thetas.shape != (2, 1)):
            return None
        x = self.add_intercept(x)
        y_hat = np.matmul(x, self.thetas)
        return y_hat
    
    def loss_elem_ (self, y, y_hat):
        print("Where is its definition?")
    
    def loss_(self, y, y_hat):
        """Computes the half mean squared error of two non-empty numpy.array,
        without any for loop.
        The two arrays must have the same dimensions.
        Args:
                y: has to be an numpy.array, a vector.
                y_hat: has to be an numpy.array, a vector.
        Returns:
                The half mean squared error of the two vectors as a float.
                None if y or y_hat are empty numpy.array.
                None if y and y_hat does not share the same dimensions.
        Raises:
                his function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0:
            return None
        if y.ndim != y_hat.ndim:
            return None
        res = 0.0
        for i in range(y.shape[0]):
            res += (y_hat[i][0] - y[i][0]) * (y_hat[i][0] - y[i][0])
        res /= 2 * y.shape[0]
        return res
    
    def mse_(y, y_hat):
        """
        Description:
                Calculate the MSE between the predicted output and the real
                output.
        Args:
                y: has to be a numpy.array, a vector of dimension m * 1.
                y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
                mse: has to be a float.
                None if there is a matching dimension problem.
        Raises:
                This function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.size == 0 or y_hat.size == 0:
            return None
        if y.ndim != y_hat.ndim:
            return None
        res = 0.0
        # print(y[1])
        for i in range(y.shape[0]):
            res += (y_hat[i] - y[i]) * (y_hat[i] - y[i])
        res /= y.shape[0]
        res = res[0]  # res seems to be an np.ndarray, don't know why
        return res