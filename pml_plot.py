
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02, labels=None):

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v') 							# markers tuple
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 			# colors tuple
	cmap = ListedColormap(colors[:len(np.unique(y))]) 				# create a colormap with a color
																	# for each outcome
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 			# determine x bounds
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 			# determine y bounds

	# numpy.arange(start, stop, step) returns an evenly spaced ndarray
	# numpy.meshgrid(arr1, arr2)

	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
		                   np.arange(x2_min, x2_max, resolution))

	# ndarray.ravel() flattens the array into a linear sequence:
	# e.g. [[1, 3], [2, 4]].ravel() = [1, 3, 2, 4]
	# ndarray.T returns the tranpose
	# so [[1, 3], [2, 4]].T.ravel() = [1, 2, 3, 4]

	# np.array([xx1.ravel(), xx2.ravel()]).T produces a list of all points on the mesh

	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)	# use perceptron to predict each point
	Z = Z.reshape(xx1.shape) 										# reshape Z according to xx1

	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
			        alpha=0.8, c=cmap(idx), 
			        label=labels[idx] if labels is not None else cl,
			        marker=markers[idx])

# EOF
