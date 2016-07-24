"""
Implementation of Adaline with stochastic gradient descent.
"""

import numpy as np

from numpy.random import seed

class AdalineSGD(object):
	"""
	ADAptive LInear NEuron classifier.

	Parameters:
	- eta: float, learning rate
	- n_iter: int, passes over the training dataset

	Attributes:
	- w_: 1d-array, weights after fitting
	- errors_: list, number of misclassifactions where index is epoch number
	- shuffle: bool (default: True), shuffles training data every epoch
	- random_state: int (default: None), set random state for shuffling
	"""

	def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle

		if random_state:
			seed(random_state)

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

		self._initialize_weights(X.shape[1])		    	# w_0, w_1, w_2, ..., w_n
		self.cost_ = [] 									# avg_cost (SSE) indexed by epoch

		for i in range(self.n_iter): 						# iterate n_iter times
			
			if self.shuffle:								# if shuffle param is True
				X, y = self._shuffle(X, y) 					# permute the data
			
			cost = [] 										# accumlator for costs

			for xi, yi in zip(X, y):						# for each observation
				cost.append(self._update_weights(xi, yi)) 	#    update the weights and append cost
			avg_cost = sum(cost) / len(y) 					# calculate the mean cost
			self.cost_.append(avg_cost) 					# append this epoch's avg_cost to cost_

		return self

	def partial_fit(self, X, y):
		"""
		Fit training data without reinitializing the weights.
		Added for on-line training of the model.
		"""

		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, yi in zip (X, y):
				self._update_weights(xi, yi)
		else:
			self._update_weights(X, y)
		return self

	def _shuffle(self, X, y):
		r = np.random.permutation(len(y))		# r is a random permutation of 0, 1, 2 ... len(y)
		return X[r], y[r]						# return the permuted data

	def _initialize_weights(self, n):
		self.w_ = np.zeros(1 + n)				
		self.w_initialized = True

	def _update_weights(self, xi, yi):
		output = self.net_input(xi)
		error = (yi - output)
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error ** 2
		return cost 							# return cost for calculation of avg cost

	def net_input(self, X):
		return X.dot(self.w_[1:]) + self.w_[0] 

	def activation(self, X):
		return self.net_input(X)				# identity function for Adaline

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1) 	# 1 if >= 0.0, -1 otherwise

# EOF
