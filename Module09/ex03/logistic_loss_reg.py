import numpy as np


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two
    non-empty numpy.ndarray, without any for l
    Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
    Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
    Raises:
            This function should not raise any Exception.
    """


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    
    # Example :
    reg_log_loss_(y, y_hat, theta, .5)
    # Output:
    0.43377043716475955
    
    # Example :
    reg_log_loss_(y, y_hat, theta, .05)
    # Output:
    0.13452043716475953
    
    # Example :
    reg_log_loss_(y, y_hat, theta, .9)
    # Output:
    0.6997704371647596