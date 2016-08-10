"""
Python Machine Learning - Chapter 3
"""

import numpy as np
import pml_plot
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression

iris = datasets.load_iris()				# import dataset
X = iris.data[:, [2, 3]] 				# project(petal length, petal width)
y = iris.target							# project(species)

print('Class labels:', np.unique(y))	# [0, 1, 2]

# create shorthand for train_test_split()

tts = lambda X, y: train_test_split(X, y, test_size=0.3, random_state=0)

X_trn, X_tst, y_trn, y_tst = tts(X, y)	# split the dataset

sc = StandardScaler()

# next two lines can be written on one line: sc.fit_transform(X_train)

sc.fit(X_trn) 							# fit() estimates the sample mean and standard deviation
X_trn_std = sc.transform(X_trn)			# transform standardizes the data
X_tst_std = sc.transform(X_tst)			# use the same mean and variance to transform test data
										# test and training data must remain comparable

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)	# instantiate a Perceptron object
ppn.fit(X_trn_std, y_trn) 								# call fit(X, y) to train algorithm
y_pred = ppn.predict(X_tst_std)							# generate a prediction vector

# in Numpy ndarrays can be compared element by element

print('Misclassified samples: %d' % (y_tst != y_pred).sum())

# do the same with sklearn.metrics instead

from sklearn.metrics import accuracy_score

# in Python 3: '%.2f' % 3.55 -> 3.5 ... '%.2f' % 3.551 -> 3.6
# so round() may be a better option
# for results of Numpy ops np.set_printoptions(precision=2)

print('Accuracy: ', round(accuracy_score(y_tst, y_pred), 2))

# an alternative without having to import from metrics:
# print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

# plot_decision_regions expects the entire dataset with
# testing following training data

# a nice introdcution to stacking methods:
# http://stackoverflow.com/questions/16473042/numpy-vstack-vs-column-stack
X_cmb_std = np.vstack((X_trn_std, X_tst_std))
y_cmb = np.hstack((y_trn, y_tst))

labels = ['Setosa', 'Versicolor', 'Virginica'] 	# added parameter for plot_decision_regions

# replace the hard-coded range for test_idx in book with something more general
# combination along with the index of the first training sample
# should be handled in a single function call: pml_plot.combined()

X_cmb_std, y_cmb, test_idx = pml_plot.combine(X_trn_std, X_tst_std, y_trn, y_tst)
pml_plot.plot_decision_regions(X_cmb_std, y_cmb, ppn, 
	                           test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.title('Iris Classification - Perceptron')
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

plt.plot(z, phi_z)											# plot sigmoid
plt.axhspan(0.0, 1.0, fc='1.0', alpha=1.0, ls='dotted')		# span between 0.0 and 1.0

# facecolor (fc): gray shades can be given as a string encoding a float (0.0 to 1.0)
#                 0.0 -> black and 1.0 -> white			 
# alpha: set the alpha tranparency
#        0.0 -> transparent and 1.0 -> opaque
# linestyle (ls): select a linestyle for the borders

plt.axvline(0.0, color='k')									# vert line at 0.0
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('$z$', fontsize=16)								# can use LaTeX formatting
plt.ylabel('$\phi (z)$', fontsize=16)
plt.title('Sigmoid Function')
plt.show()

# *** consider generating the plot shown plotting J(w) as a function of Phi(z)
# as shown in text, but not coded

lr = LogisticRegression(C=1000.0, random_state=0)			# instantiate a LR classifier 
lr.fit(X_trn_std, y_trn) 									# train the model

pml_plot.plot_decision_regions(X_cmb_std, y_cmb, lr,
	                           test_idx=test_idx, labels=labels)
plt.xlabel('Petal Length [standardized]')
plt.ylabel('Petal Width [standardized]')
plt.title('Iris Classification - Logistic Regression')
plt.legend(loc='upper left')
plt.show()

# for an explaination of numpy set_printoptions kwargs: 
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
np.set_printoptions(precision=3, suppress=True)

print(lr.predict_proba(X_tst_std[0, :].reshape(1, -1)))

# output without first setting the formatting options:
# [[  2.05743774e-11   6.31620264e-02   9.36837974e-01]]

# output after using set_printoptions:
# [[ 0.     0.063  0.937]]

weights, params = [], [] 								# accumulators

for c in np.arange(-5, 5):
	lr = LogisticRegression(C=10**c, random_state=0)
	lr.fit(X_trn_std, y_trn)
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

# from sklearn.svm import SVC

# svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm.fit(X_trn_std, y_trn)

# pml_plot.plot_decision_regions(X_cmb_std,
# 							   y_cmb, svm,
# 							   test_idx=test_idx,
# 							   labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# """
# If the dataset is too large to fit into main memory
# then we need an online solution. The SGDClassifier
# types support the partial_fit method and a stochastic
# approach to updating the model.
# """

# from sklearn.linear_model import SGDClassifier

# ppn = SGDClassifier(loss='perceptron')
# lr = SGDClassifier(loss='log')
# svm = SGDClassifier(loss='hinge')

# # handling nonlinear classification tasks

# np.random.seed(0)
# X_xor = np.random.randn(200, 2)
# y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
# 	        c='b', marker='x', label='1')
# plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
# 	        c='r', marker='s', label='-1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()		# non-linearly separable data

# svm = SVC(kernel='rbf', gamma=0.10, C=10.0, random_state=0)
# svm.fit(X_xor, y_xor)
# pml_plot.plot_decision_regions(X_xor, y_xor, svm)
# plt.legend(loc='upper left')
# plt.show()

# # rbf applied to iris dataset

# svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)	# small gamma -> soft decision boundry
# svm.fit(X_trn_std, y_trn)
# pml_plot.plot_decision_regions(X_cmb_std,
# 							   y_cmb, svm,
# 							   test_idx=test_idx,
# 							   labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0) 	# large gamma -> hard decision boundry
# svm.fit(X_trn_std, y_trn)
# pml_plot.plot_decision_regions(X_cmb_std,
# 							   y_cmb, svm,
# 							   test_idx=test_idx,
# 							   labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# # decision trees
# # do all of these assume two classes?

# def gini(p):
# 	# return p * (1 - p) + (1 - p) * (1 - (1 - p))
# 	return 2 * (p - p**2)	# simplified formula for two classes

# def entropy(p):
# 	return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# def error(p):
# 	return 1 - np.max([p, 1 - p])

# x = np.arange(0.0, 1.0, 0.01)
# ent = [entropy(p) if p != 0 else None for p in x]
# sc_ent = [e*0.5 if e else None for e in ent]
# err = [error(i) for i in x]
# fig = plt.figure()
# ax = plt.subplot(111)
# for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
# 	                      ['Entropy', 'Entropy (scaled)',
# 	                       'Gini Impurity', 'Misclassification Error'],
# 	                      ['-', '-', '--', '-.'],
# 	                      ['black', 'lightgray', 'red', 'green', 'cyan']):
# 	line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
# 	       ncol=3, fancybox=True, shadow=False)
# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# plt.ylim([0, 1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('Impurity Index')
# plt.show()

# # growing a tree

# from sklearn.tree import DecisionTreeClassifier

# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# tree.fit(X_trn, y_trn)	# no need to stdize for tree methods

# X_cmb = np.vstack((X_trn, X_tst))
# y_cmb = np.hstack((y_trn, y_tst))

# pml_plot.plot_decision_regions(X_cmb, y_cmb, tree,
# 	                           test_idx=test_idx, labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# # make use of GraphViz

# from sklearn.tree import export_graphviz

# export_graphviz(tree, out_file='tree.dot',
# 	            feature_names=['Petal Length', 'Petal Width'])

# # random forest

# from sklearn.ensemble import RandomForestClassifier

# forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
# 	                            random_state=1, n_jobs=2) 	# n_jobs uses two cores
# forest.fit(X_trn, y_trn)

# pml_plot.plot_decision_regions(X_cmb, y_cmb, forest,
# 							   test_idx=test_idx, labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# # knn

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# knn.fit(X_trn_std, y_trn)

# pml_plot.plot_decision_regions(X_cmb_std, y_cmb, knn,
# 							   test_idx=test_idx, labels=labels)
# plt.xlabel('Petal Length [standardized]')
# plt.ylabel('Petal Width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# EOF
