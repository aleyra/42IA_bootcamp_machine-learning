import numpy as np
import random


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a
    test set, while respecting the given proportion of examples to be kept in
    the training set.
    Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
            proportion: has to be a float, the proportion of the dataset that
            will be assigned to the training set.
    Return:
            (x_train, x_test, y_train, y_test) as a tuple of numpy.array
            None if x or y is an empty numpy.array.
            None if x and y do not share compatible dimensions.
            None if x, y or proportion is not of expected type.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(proportion, float)
    ):
        return None
    if proportion >= 1 or proportion <= 0 or x.size == 0 or y.size == 0:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    
    #init
    size_train = int(float(x.shape[0]) * proportion)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # get list of indice of row for train
    split_train = random.sample(range(x.shape[0]), size_train)
    
    # get list of indice not in train
    split_test = []
    for i in range(x.shape[0]):
        if i not in split_train:
            split_test.append(i)
    
    # shuffle test
    split_test = random.sample(split_test, len(split_test))
    # print(f"train = {split_train}\ntest = {split_test}")
    
    # split datas in train and test in shuffle way
    for i in split_train:
        # print(f"x[{i}] = {x[i]} et y[{i}] = {y[i]}")
        x_train = np.append(x_train, x[i], axis = 0)
        y_train = np.append(y_train, y[i], axis = 0)
    for i in split_test:
        # print(f"x[{i}] = {x[i]} et y[{i}] = {y[i]}")
        x_test = np.append(x_test, x[i], axis = 0)
        y_test = np.append(y_test, y[i], axis = 0)

    x_train = x_train.reshape((size_train, x.shape[1]))
    x_test = x_test.reshape((x.shape[0] - size_train, x.shape[1]))
    y_train = y_train.reshape((size_train, y.shape[1]))
    y_test = y_test.reshape((y.shape[0] - size_train, y.shape[1]))
    data_split = (x_train, x_test, y_train, y_test)
    return data_split


if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

    # print(x1)

    # Example 1:
    ds = data_spliter(x1, y, 0.8)
    print(f"""exemple 1
    x_train = {ds[0]}
    x_test = {ds[1]}
    y_train = {ds[2]}
    y_test = {ds[3]}""")
    # Output:
    # (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))

    # Example 2:
    ds = data_spliter(x1, y, 0.5)
    print(f"""exemple 2
    x_train = {ds[0]}
    x_test = {ds[1]}
    y_train = {ds[2]}
    y_test = {ds[3]}""")
    # Output:
    # (array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1]))

    x2 = np.array([[ 1, 42],
        [300, 10],
        [ 59, 1],
        [300, 59],
        [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

    # Example 3:
    ds = data_spliter(x2, y, 0.8)
    print(f"""exemple 3
    x_train = {ds[0]}
    x_test = {ds[1]}
    y_train = {ds[2]}
    y_test = {ds[3]}""")
    # Output:
    # (array([[ 10, 42],
    #     [300, 59],
    #     [ 59, 1],
    #     [300, 10]]),
    # array([[ 1, 42]]),
    # array([0, 1, 0, 1]),
    # array([0]))

    # Example 4:
    ds = data_spliter(x2, y, 0.5)
    print(f"""exemple 4
    x_train = {ds[0]}
    x_test = {ds[1]}
    y_train = {ds[2]}
    y_test = {ds[3]}""")
    # Output:
    # (array([[59, 1],
    #     [10, 42]]),
    # array([[300, 10],
    #     [300, 59],
    #     [ 1, 42]]),
    # array([0, 0]),
    # array([1, 1, 0]))