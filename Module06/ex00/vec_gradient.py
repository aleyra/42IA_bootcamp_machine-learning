import numpy as np


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
                ction should not raise any Exception.
    """
