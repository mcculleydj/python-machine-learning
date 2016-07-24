"""
Title:       linear_regression.py
Date:        7/12/16
Author:      Jaques Grobler
License:     BSD 3
Modified by: Jared Bronen and Daren McCulley
"""

"""
Code was only modified to switch the features to sepal lenth and width
and provide information as prompted in the assignment
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the boston dataset

boston = datasets.load_boston()

# Project for first and third attribute

X = boston.data[:, 0:2]
y = boston.target

# Split the data into training/testing sets
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

# Create linear regression object

regr = linear_model.LinearRegression()

# Train the model using the training sets

regr.fit(X_train_std, y_train)

# The coefficients
print('Coefficients:', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test_std) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test_std, y_test))

# Predictions
print('Predictions:')
predictions = regr.predict(X_unk_std)

for i in range(3):
    print('x_%s: %s' % (i, X_unk[i]), end=' ')
    print('y_%s: %s' % (i, predictions[i]))
print()

y_pred = regr.predict(X_test_std)

from sklearn import metrics

MSE = metrics.mean_squared_error(y_test, y_pred)
print('MSE: %s' % MSE)

# Alternate method... same result as .score()

R2 = metrics.r2_score(y_test, y_pred)
print('R2: %s' % R2)

# EOF
