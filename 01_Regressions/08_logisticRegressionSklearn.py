
"""To Remember

nothing. Actually, it's really easy to use Logistic Regression in SciKit-Learn.

"""

## Logistic Regression Using SciKit-Learn
## ________________________________________START_________________________________________________

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# fit the model
lr_model = LogisticRegression()
lr_model.fit(X, y)

# predict
y_pred = lr_model.predict(X)
print(y_pred)

# calculate the accuracy
print(f"Accuacy on training set: {lr_model.score(X, y)}")