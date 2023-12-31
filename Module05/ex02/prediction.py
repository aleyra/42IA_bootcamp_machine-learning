import numpy as np

def predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.array.
		Args:
			x: has to be an numpy.array, a vector of dimension m * 1.
			theta: has to be an numpy.array, a vector of dimension 2 * 1.
		Returns:
			y_hat as a numpy.array, a vector of dimension m * 1.
			None if x and/or theta are not numpy.array.
			None if x or theta are empty numpy.array.
			None if x or theta dimensions are not appropriate.
		Raises:
			function should not raise any Exceptions.
	"""
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
		return None
	if (x.size == 0 or theta.size == 0):
		return None
	if (x.ndim != 1
     	or theta.shape[0] != 2 or theta.shape[1] != 1):
		return None
	t = []
	nb_line = x.shape[0]
	for i in range(nb_line):
		t.append(1)
		t.append(x[i])
	t = np.array(t)
	x = t.reshape(nb_line, 2)
	# print(x)
	y_hat = np.matmul(x, theta)
	# y_hat = None
	return y_hat

if __name__ == "__main__":
	x = np.arange(1,6)

	# Example 1:
	theta1 = np.array([[5], [0]])
	# print(predict_(x, theta1))
	# Ouput:
	# print(np.array([[5.], [5.], [5.], [5.], [5.]]))
	# Do you remember why y_hat contains only 5’s here?

	# Example 2:
	theta2 = np.array([[0], [1]])
	# print(predict_(x, theta2))
	# Output:
	# array([[1.], [2.], [3.], [4.], [5.]])
	# Do you remember why y_hat == x here?

	# Example 3:
	theta3 = np.array([[5], [3]])
	# print(predict_(x, theta3))
	# Output:
	# array([[ 8.], [11.], [14.], [17.], [20.]])

	# Example 4:
	theta4 = np.array([[-3], [1]])
	print(predict_(x, theta4))
	# Output:
	# array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])