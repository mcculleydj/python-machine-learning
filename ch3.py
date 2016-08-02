"""
Python Machine Learning - Chapter 3
"""

import numpy as np
import pml_plot
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()				# import dataset
X = iris.data[:, [2, 3]] 				# project(petal length, petal width)
y = iris.target							# project(species)

print('Class labels:', np.unique(y))	# [0, 1, 2]

# create shorthand for train_test_split()

tts = lambda X, y: train_test_split(X, y, test_size=0.3, random_state=0)

X_train, X_test, y_train, y_test = tts(X, y)	# split the dataset

sc = StandardScaler()

# next two lines can be written on one line: sc.fit_transform(X_train)

sc.fit(X_train) 						# fit() estimates the sample mean and standard deviation
X_train_std = sc.transform(X_train)		# transform standardizes the data
X_test_std = sc.transform(X_test)		# use the same mean and variance to transform test data
										# test and training data must remain comparable

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)	# instantiate a Perceptron object
ppn.fit(X_train_std, y_train) 							# call fit(X, y) to train algorithm
y_pred = ppn.predict(X_test_std)						# generate a prediction vector

# in Numpy ndarrays can be compared element by element

print('Misclassified samples: %d' % (y_test != y_pred).sum())

# do the same with sklearn.metrics instead

from sklearn.metrics import accuracy_score

# in Python 3: '%.2f' % 3.55 -> 3.5 ... '%.2f' % 3.551 -> 3.6
# so round() may be a better option
# for results of Numpy ops np.set_printoptions(precision=2)

print('Accuracy: ', round(accuracy_score(y_test, y_pred), 2))

# an alternative without having to import from metrics:
# print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

# plot_decision_regions expects the entire dataset with
# testing following training data

# a nice introdcution to stacking methods:
# http://stackoverflow.com/questions/16473042/numpy-vstack-vs-column-stack
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

labels = ['Setosa', 'Versicolor', 'Virginica'] 	# added parameter for plot_decision_regions

# replace the hard-coded range for test_idx with something more general
# combination along with the index of the first training sample
# should be handled in a single function call: pml_plot.combined()

X_combined_std, y_combined, test_idx = pml_plot.combine(X_train_std, X_test_std, y_train, y_test)
pml_plot.plot_decision_regions(X_combined_std, y_combined, ppn, 
	                           test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
odds ratio: p / (1 - p), where p is the probability of an event happening
logit function: log (p / (1 - p)), simply the log of the odds ratio or log-odds
"""

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi $ (z)')
plt.show()

# consider generating the plot shown plotting J(w) as a function of Phi(z)
# as shown in text, but not coded

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
pml_plot.plot_decision_regions(X_combined_std, y_combined, lr, test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

print(np.around(lr.predict_proba(X_test_std[0,:].reshape(1, -1)), 3))

weights, params = [], []

for c in np.arange(-5, 5):
	lr = LogisticRegression(C=10**c, random_state=0)
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)
weights = np.array(weights)

plt.plot(params, weights[:, 0], linestyle='-', label='Petal Length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal Width')
plt.ylabel('Weight Coeff')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

pml_plot.plot_decision_regions(X_combined_std,
							   y_combined, svm,
							   test_idx=test_idx,
							   labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
If the dataset is too large to fit into main memory
then we need an online solution. The SGDClassifier
types support the partial_fit method and a stochastic
approach to updating the model.
"""

from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

# handling nonlinear classification tasks

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
	        c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
	        c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()		# non-linearly separable data

svm = SVC(kernel='rbf', gamma=0.10, C=10.0, random_state=0)
svm.fit(X_xor, y_xor)
pml_plot.plot_decision_regions(X_xor, y_xor, svm)
plt.legend(loc='upper left')
plt.show()

# rbf applied to iris dataset

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)	# small gamma -> soft decision boundry
svm.fit(X_train_std, y_train)
pml_plot.plot_decision_regions(X_combined_std,
							   y_combined, svm,
							   test_idx=test_idx,
							   labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0) 	# large gamma -> hard decision boundry
svm.fit(X_train_std, y_train)
pml_plot.plot_decision_regions(X_combined_std,
							   y_combined, svm,
							   test_idx=test_idx,
							   labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

# decision trees
# do all of these assume two classes?

def gini(p):
	# return p * (1 - p) + (1 - p) * (1 - (1 - p))
	return 2 * (p - p**2)	# simplified formula for two classes

def entropy(p):
	return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
	return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
	                      ['Entropy', 'Entropy (scaled)',
	                       'Gini Impurity', 'Misclassification Error'],
	                      ['-', '-', '--', '-.'],
	                      ['black', 'lightgray', 'red', 'green', 'cyan']):
	line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
	       ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

# growing a tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)	# no need to stdize for tree methods

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

pml_plot.plot_decision_regions(X_combined, y_combined, tree,
	                           test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

# make use of GraphViz

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree.dot',
	            feature_names=['Petal Length', 'Petal Width'])

# random forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
	                            random_state=1, n_jobs=2) 	# n_jobs uses two cores
forest.fit(X_train, y_train)

pml_plot.plot_decision_regions(X_combined, y_combined, forest,
							   test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

# knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

pml_plot.plot_decision_regions(X_combined_std, y_combined, knn,
							   test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.legend(loc='upper left')
plt.show()

# EOF
