

""" To Remember

1. SGDRegressor acts like the gradient descent algorithm.
    - sgdr = SGDRegressor(max_iter=1000)
    - sgdr.fit(X_train, y_train)
    - b_norm = sgdr.intercept_
    - w_norm = sgdr.coef_
    - predictions = sgdr.predict(X_train)
2. ScandardScaler is used to normalize the features.
    - scaler = StandardScaler()
    - X_train = scaler.fit_transform(X_train)

"""


# ________________________________________START____________________________________________________
import numpy as np
from sklearn.linear_model import SGDRegressor # Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler # z-score normalization

# load the dataset
X_train, y_train = (np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]), np.array([460, 232, 178]))

# feature scaling by z-score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# fit the model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train, y_train)
# Note: sgdr.fit always receives a 2d matrix as X_train data.

# view parameters
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

# making predictions
predictions = sgdr.predict(X_train)
predictions2 = np.dot(w_norm, X_train.T) + b_norm
# X_Train.T is the transpose of X_train. It is used to make the dot product work.



