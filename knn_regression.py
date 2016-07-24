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

boston = datasets.load_boston()
X = boston.data[:, [0,2]]
y = boston.target

# Splitting data into 70% training and 30% test data

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_unk = np.array([[.02, 2], [.025, 13], [.015, 25]])

# Normalize the features

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_unk_std = sc.transform(X_unk)
                
# KNN-Regression

from sklearn.neighbors import KNeighborsRegressor

knnr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
knnr.fit(X_train_std, y_train)
results = knnr.score(X_test_std, y_test)

print('Confidence score: %s\n' % results)
print('Predictions for X_unk:')

predictions = knnr.predict(X_unk_std)

for i in range(3):
    print('x_%s: %s' % (i, X_unk[i]), end=' ')
    print('y_%s: %s' % (i, predictions[i]))
print()

y_pred = knnr.predict(X_test_std)

from sklearn import metrics

MSE = metrics.mean_squared_error(y_test, y_pred)
print('MSE: %s' % MSE)

# Alternate method... same result as .score()

R2 = metrics.r2_score(y_test, y_pred)
print('R2: %s' % R2)

# EOF

