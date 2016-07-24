"""
Title:       knn.py
Date:        7/12/16
Author:      Gautam Bhat
Modified by: Jared Bronen and Daren McCulley
"""

"""
Code was only modified to switch the features to sepal lenth and width
and provide information as prompted in the assignment
"""

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [0,1]]
y = iris.target

print('Class labels:', np.unique(y))

# Splitting data into 70% training and 30% test data

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# X inputs for which y is unknown

X_unk = np.array([[5.5, 4.0], [6.0, 2.3], [4.5, 2.0]])

# Normalize the features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_unk_std = sc.transform(X_unk)

# Plotting methods

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
                                        
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
results = knn.score(X_test_std, y_test)

print('Confidence score: %s\n' % results)
print('Predictions for X_unk:')

predictions = knn.predict(X_unk_std)

for i in range(3):
    print('x_%s: %s' % (i, X_unk[i]), end=' ')
    print('y_%s: %s' % (i, predictions[i]))
print()

# Plot

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel('sepal length [standardized]')
plt.ylabel('sepal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
plt.show()

# EOF
