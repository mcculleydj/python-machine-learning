"""
Implementation of the examples in Chapter 2
"""

import numpy as np
import pml_plot
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train) 						# fit() estimates the sample mean and standard deviation
X_train_std = sc.transform(X_train)		# transform standardizes the data
X_test_std = sc.transform(X_test)		# same mean and variance used to transform test data
										# test and training data must remain comparable

from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
labels = ['Setosa', 'Versicolor', 'Virginica']
pml_plot.plot_decision_regions(X_combined_std, y_combined, ppn, test_idx=range(105, 150), labels=labels)
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

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
pml_plot.plot_decision_regions(X_combined_std, y_combined, lr, test_idx=range(105, 150), labels=labels)
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














