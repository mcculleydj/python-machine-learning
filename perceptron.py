"""
The percepton classification method determines the 
n+1 coefficients which define an n-dimensional hyperplane.
The goal is to divide the training set {(x1, y1), ... (xm, ym)},
where each yi is one of two classes, such that all observations
from the same class are on the same side of the linear split.

If the training set can be divided linearly then this alogithm 
will converge. However, if the set cannot be split by a 
hyperplane such that each half is homogeneous then this 
algorithm will never terminate. This is why the parameters:
n_iter and eta are critical.

n_iter bounds the number of iterations. 
Eta is the learning rate, which adjusts the magnitude 
of the correction to the weights (coefs) during each iteration.
"""

import numpy as np

class Perceptron(object):

	"""
	parameters for constructor: eta, n_iters
	fields: w_, errors_
	methods: fit(X, y), net_input(X), predict(X)
	"""

	def __init__(self, eta, n_iter):
		self.eta = eta						# eta is the learning rate 0.0 < eta < 1.0
		self.n_iter = n_iter 				# n_iter is the number of iterations
	
	# X is an m by n matrix where:
	#   m is the number of observations
	# 	n is the number of features for each observation

	# y is an m-element vector of target values for each x in X

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1]) 	# initialize weight vector of all zeros 
		                                    # s.t. length = 1 + X.shape[1] (1 + number of features)
		                                    
		self.errors_ = [] 					# initialize errors_ to an empty list

		for _ in range(self.n_iter): 							# execute loop n_iter times
			errors = 0 											#   set errors to 0 (counter)
			for xi, yi in zip(X, y): 							#   for xi, yi in training set
				update = self.eta * (yi - self.predict(xi))		#      set update to eta * error
																#      where error = yi - self.predict(xi)
				self.w_[1:] += update * xi 						#      w_[1:] += update * xi
				self.w_[0] += update 							#      w_[0] += update
				errors += int(update != 0.0) 					#      errors += 0 or 1
																#      if update == 0.0 then 0, else 1
																#      update == 0.0 iff error == 0.0
			self.errors_.append(errors)							#   append errors to errors_
			if errors == 0:										#   if current weights result in no errors
				return self										#      return self
		return self 											# return self

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0] 			# return sum(w0 + w1 * x1 + ... + wn * xn)

	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)	# return 1 if net_input(X) >= 0.0
															# return -1 otherwise

# EOF
