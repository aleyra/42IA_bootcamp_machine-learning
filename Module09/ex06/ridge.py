import numpy as np
from my_linear_regression import MyLinearRegression as MLR
from l2_reg import l2
from vec_log_loss import vec_log_loss_ as v_log_loss_


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

    def gradient_(self, ):

    def fit_(self, ):
