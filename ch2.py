"""
Implementation of the examples in Chapter 2
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

# dataset attributes:
# 0 - sepal length in cm 
# 1 - sepal width in cm 
# 2 - petal length in cm 
# 3 - petal width in cm 
# 4 - class: {Iris-setosa, Iris-versicolour, Iris-virginica}

y = df.iloc[0:100, 4].values 				# select the first 100 rows and project on class
y = np.where(y == 'Iris-setosa', -1, 1)		# map 'Iris-setosa' to -1 and 'Iris-versicolor' to 1
X = df.iloc[0:100, [0, 2]].values			# select the first 100 rows and 
											# project on sepal length and petal length
# data is pre-sorted on class
# - the first 50 rows are setosa
# - the next 50 rows are versicolor

# plot the data

plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='Setosa') 		# plot setosa pts
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='Versicolor') 	# plot versicolor pts
plt.xlabel('Sepal Length [cm]')
plt.ylabel('Petal Length [cm]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title('Iris Data')
plt.show()

# train perceptron classifier

ppn = perceptron.Perceptron(eta=0.1, n_iter=10) 		# instantiate a perceptron object
ppn.fit(X, y) 											# call fit(X, y) to train algorithm

# plot number of errors at each epoch

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlim([1, len(ppn.errors_)])							# assume epochs start at 1
plt.ylim([0, max(ppn.errors_) + 1])						# use max to bound the axis
plt.xticks(np.arange(1, len(ppn.errors_) + 1, 1))		# +1 because range is non-incl
plt.yticks(np.arange(0, max(ppn.errors_) + 2, 1))		# +2 because non-incl and want 1 higher
plt.title('Convergence of Percepton')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifcations')
plt.show()

# plot decision regions

labels = ['Setosa', 'Versicolor']
pml_plot.plot_decision_regions(X, y, classifier=ppn, labels=labels)
plt.title('Perceptron Decision Regions')
plt.xlabel('Sepal Length [cm]')
plt.ylabel('Petal Length [cm]')
plt.legend(loc='upper left')
plt.show()

# Adaline

# create a figure for two plots

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
# ax is a (2,) ndarray containing two axis objects

# ada1 will diverge because eta is too large

ada1 = adaline_gd.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# ada2 will converge on a solution

ada2 = adaline_gd.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()

# we can prevent divergence by standardizing the training set
# this prevents the scale of variables from impacting the algorithm
# see sklearn standardization methods for an off-the-shelf method
# which is used in ch3.py

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

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
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Convergence of Adaline w/ Batch GD')
plt.show()

# AdalineSGD

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
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Convergence Stochastic GD')
plt.show()

# EOF
