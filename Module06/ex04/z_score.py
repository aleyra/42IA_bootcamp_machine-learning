import numpy as np
import math
from TinyStatistician import TinyStatistician

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the
    z-score standardization.
    Args:
            x: has to be an numpy.ndarray, a vector.
    Returns:
            x_prime as a numpy.ndarray.
            None if x is not a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
            This function shouldn't raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    
    tstat = TinyStatistician()
    mean = tstat.mean(x)

    std = 0.0
    for i in range(x.shape[0]):
        std += (x[i] - mean) * (x[i] - mean)
    std = std / x.shape[0]
    std = math.sqrt(std)  #french version of std -> same result but different formula
    # std = tstat.std(x)  #english version of std -> different result but same formula
    x_prime = np.ndarray(x.shape)
    for i in range(x.shape[0]):
        x_prime[i] = (x[i] - mean) / std
    return x_prime

if __name__ == "__main__":
    # Example 1:
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    # Output:
    # array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559,
    # 0.17240647, -1.89647119])

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y))
    # Output:
    # array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
    # 0.28795027, -1.72770165])
