import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
            y_true: numpy.ndarray for the correct labels
            y_hat: numpy.ndarray for the predicted labels
            labels: Optional, a list of labels to index the matrix.
            This may be used to reorder or select a subset of labels. (default=None)
    Returns:
            The confusion matrix as a numpy ndarray.
            None on any error.
    Raises:
            This function should not raise any Exception.
    """
    ... Your code ...


if __name__ == "__main__":
    y_hat = np.array(
        [['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']]
    )
    y = np.array(
        [['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']]
    )
    # Example 1:
    ## your implementation
    confusion_matrix_(y, y_hat)
    ## Output:
    # array([[0 0 0]
    #     [0 2 1]
    #     [1 0 2]])
    ## sklearn implementation
    confusion_matrix(y, y_hat)
    ## Output:
    # array([[0 0 0]
    #     [0 2 1]
    #     [1 0 2]])

    # Example 2:
    ## your implementation
    confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])
    ## Output:
    # array([[2 1]
    #     [0 2]])
    ## sklearn implementation
    confusion_matrix(y, y_hat, labels=['dog', 'norminet'])
    ## Output:
    # array([[2 1]
    #     [0 2]])