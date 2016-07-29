"""
Python Machine Learning - Chapter 2
"""

import pml_plot
import perceptron
import adaline_gd
import adaline_sgd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

# import the iris data using pandas into a DataFrame object
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)

# iris dataset attributes:
# 0 - sepal length in cm 
# 1 - sepal width in cm 
# 2 - petal length in cm 
# 3 - petal width in cm 
# 4 - species: {Iris-setosa, Iris-versicolour, Iris-virginica}

# data is pre-sorted by species
# - the first 50 rows are setosa
# - the next 50 rows are versicolor

X = df.iloc[0:100, [0, 2]].values			# select(setosa, versicolor) and project(sepal length, petal length)											
y = df.iloc[0:100, 4].values 				# select(setosa, versicolor) and project(species)
y = np.where(y == 'Iris-setosa', -1, 1)		# map 'Iris-setosa' to -1 and 'Iris-versicolor' to 1

# plot the data

plt.scatter(X[:50, 0], X[:50, 1], 
            c='r', marker='o', label='Setosa') 			# plot setosa observations
plt.scatter(X[50:100, 0], X[50:100, 1],
            c='b', marker='x', label='Versicolor') 		# plot versicolor observations
ax = plt.gca()											# get the current Axes instance on the current figure 
														# matching kwargs, or create one.
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))	# setting the format to int for the xaxis
plt.xticks(np.arange(4, 9, 1))							# match x-axis ticks to data
plt.title('Iris Classification by Sepal Length and Petal Length')
plt.xlabel('Sepal Length [cm]')
plt.ylabel('Petal Length [cm]')
plt.legend(loc='upper left')
plt.show()

# train perceptron classifier

ppn = perceptron.Perceptron(eta=0.1, n_iter=10) 	# instantiate a perceptron object
ppn.fit(X, y) 										# call fit(X, y) to train algorithm

# plot number of errors at each epoch

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlim([1, len(ppn.errors_)])						# x-axis from 1 to number of epochs
plt.ylim([0, max(ppn.errors_) + 1])					# y-axis from 0 to max number of errors + 1
plt.xticks(np.arange(1, len(ppn.errors_) + 1, 1))	# match x-axis ticks to xlim
plt.yticks(np.arange(0, max(ppn.errors_) + 2, 1))	# match y-axis ticks to ylim
plt.title('Convergence of Percepton')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifcations')
plt.show()

# plot decision regions

labels = ['Setosa', 'Versicolor']					# parameter for plot_decision_regions
pml_plot.plot_decision_regions(X, y, classifier=ppn, labels=labels)
plt.title('Perceptron Decision Regions')
plt.xlabel('Sepal Length [cm]')
plt.ylabel('Petal Length [cm]')
plt.legend(loc='upper left')
plt.show()

# obtain a reference to a figure with two subplots
# ax is a (2,) ndarray containing two Axes objects (one for each subplot)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

# ada1 diverges because eta is too large

ada1 = adaline_gd.AdalineGD(n_iter=10, eta=0.01).fit(X, y)	# train an Adaline_GD object
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_title('Adaline - Learning rate 0.01')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')

# ada2 converges because eta is small enough

ada2 = adaline_gd.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_title('Adaline - Learning rate 0.0001')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')

plt.show()

# standardizing the training set can prevent divergence because
# the impact of each variable is decoupled from its scale
# see sklearn standardization methods for library methods
# used in Chapter 3

X_std = np.copy(X) 	# copy values, not the reference
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()	# standard score: (x_i - mean) / std_dev
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()	# for all x_i in X

ada = adaline_gd.AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)

pml_plot.plot_decision_regions(X_std, y, classifier=ada, labels=labels)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length [standardized]')
plt.ylabel('Petal Length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlim([1, len(ada.cost_)])
plt.ylim([0, max(ada.cost_) + 1])
plt.xticks(np.arange(1, len(ada.cost_) + 1, 1))
plt.title('Convergence of Adaline_BGD (standardized)')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

# stochastic gradient descent

ada = adaline_sgd.AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)

pml_plot.plot_decision_regions(X_std, y, classifier=ada, labels=labels)
plt.title('Adaline w/ Stochastic GD - Decision Regions')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlim([1, len(ada.cost_)])
plt.xticks(np.arange(1, len(ada.cost_) + 1, 1))
plt.title('Convergence Stochastic GD (standardized)')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# EOF
