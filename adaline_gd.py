import numpy as np

class AdalineGD(object):
	"""
	ADAptive LInear NEuron classifier.

	Parameters:
	- eta: float, learning rate (0.0 to 1.0)
	- n_iter: int, passes over the training dataset

	Attributes:
	- w_: 1d-array, weights after fitting
	- errors_: list, number of misclassifactions where index is epoch number
	"""

	def __init__(self, eta=0.01, n_iter=50):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""
		Fit training data.

		Parameters:
		- X: array-like, shape = [n_samples, n_features]
		     matrix of training vectors
		- y: array-like, shape = [n_samples]
		     target values corresponding to rows in X

		Returns:
		- self: object
		"""

		self.w_ = np.zeros(1 + X.shape[1])				# w_0, w_1, w_2, ..., w_n
		self.cost_ = [] 								# value of cost function at each iteration

		for i in range(self.n_iter): 					# iterative learning algorithm
			output = self.net_input(X) 					# predicted value vector
			errors = (y - output) 						# error vector

			self.w_[1:] += self.eta * X.T.dot(errors) 	# update w_1 through w_n
			self.w_[0] += self.eta * errors.sum() 		# update w_0

			cost = (errors ** 2).sum() / 2.0 			# SSE

			self.cost_.append(cost) 					# append cost for this iteration
		return self

	def net_input(self, X):
		return X.dot(self.w_[1:]) + self.w_[0] 			# X (size: m x n) dot w_ (size: n) + w_0

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X): # is 0.0 the threshold for Adaline then? why?
		"""
		Return class label after unit step.
		"""

		return np.where(self.activation(X) >= 0.0, 1, -1) 	# 1 if >= 0.0, -1 otherwise

# EOF
