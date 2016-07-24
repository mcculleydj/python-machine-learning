"""
Implementation of Sequential Backward Selection
Python Machine Learning Ch 4
"""

import numpy as np

from sklearn.base import clone
from itertools import combinations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
	def __init__(self, estimator, k_features, test_size=0.25,
		         scoring=accuracy_score, random_state=1):
		self.estimator    = clone(estimator) 	# constructs a new estimator with the same parameters (deep copy)
		self.k_features   = k_features 			# desired number of features
		self.test_size    = test_size 			# percentage of data to split off for testing
		self.scoring      = scoring 			# scoring metric
		self.random_state = random_state 		# input used for random sampling by test_train_split()

	def fit(self, X, y):
		"""
		X is an m * n matrix and y is an m element vector
		m: number of observations
		n: number of features per observation
		Note that X_train is passed to this function and not the original X.
		In other words, the test set (validation set) is not used to perform SBS.
		This function goes onto split X_train into training and testing sets for SBS.
		"""

		# split data into training and testing for SBS

		X_train, X_test, y_train, y_test = (
			train_test_split(X, y, test_size=self.test_size, 
				             random_state=self.random_state)
										   )

		dim = X_train.shape[1] 					# number of features
		self.indices_ = tuple(range(dim)) 		# (0, 1, 2, ..., m-1)
		self.subsets_ = [self.indices_] 		# [(0, 1, 2, ..., m-1)]
		
		# score the original set containing all n features

		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		
		self.scores_ = [score] 					# instantiate scores_ as a list with m-score

		while dim > self.k_features: 			# while d > k (there are still dimensions to remove)
			scores  = [] 						# 	accumulate potential scores
			subsets = [] 						# 	accumulate potential subsets

			# for each subset of size dim-1 (combinations returns all r-element subsets)

			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)	# calc score for this combination
				scores.append(score) 											# append score to scores
				subsets.append(p) 												# append p to subsets

			best = np.argmax(scores) 			#	a maximum accuracy score determines the best subset
			self.indices_ = subsets[best]		#	append the best subset to indices_
			self.subsets_.append(self.indices_)	#	append the best subset to subsets_
			dim -= 1 							#	decrement dim
			self.scores_.append(scores[best]) 	# 	append the best score to scores_

		self.k_score_ = self.scores_[-1] 		# upon reaching k features, report the last score
		return self 							# return sbs instance after fitting

	def transform(self, X):
		return X[:, self.indices_]				# projection to retain only the columns in self.indices_

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train[:, indices], y_train) 		# use the estimator to fit X with columns in indices
		y_pred = self.estimator.predict(X_test[:, indices]) 	# use the estimator to generate y_pred vector
		score = self.scoring(y_test, y_pred) 					# use the scoring method to evaluate model
		return score

# EOF
