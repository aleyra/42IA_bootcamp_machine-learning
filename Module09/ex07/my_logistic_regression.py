import numpy as np


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    supported_penalities = ['l2']
    # We consider l2 penality only. One may wants to implement other penalities
    def __init__(
            self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0
    ):
    # Check on type, data type, value ... if necessary
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0
    #... Your code ...
    #... other methods ...