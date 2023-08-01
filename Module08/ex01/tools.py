import numpy as np

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x.
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
	res = np.insert(x, 0, col1, axis = 1)
	return res

def from_shape(shape, value = 0):
	print(shape)
	nb = 1
	for i in range(len(shape)):
		nb *= shape[i]
	t = []
	for i in range(nb):
		t.append(value)
	t = np.array(t)
	return t.reshape(shape)

if __name__ == "__main__":
	shape = (2, 3)
	arr = from_shape(shape, 2)
	print(add_intercept(arr))