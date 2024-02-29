
""" To remember

there three types of feature scaling:
1. z-score (standardization)
2. min-max scaling
3. mean normalization

how np.mean, np.std, np.min, np.max work.
what is meant by axis=0 and axis=1 in np.mean, np.std, np.min, np.max

"""
# ________________________________________START____________________________________________________


import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


""" feature scaling by z-score 

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_mean = np.mean(X_train, axis=0)

X_norm = (X_train - X_mean) / sigma

# axis=0 means column-wise. axis=1 means row-wise. In 3rd dimension, axis=0 is depth-wise. 
# (The pattern is that the axis is the dimension that will be collapsed)

what happens if we average by axis=1?
a = np.array([[1, 2, 3], [4, 5, 6],[4, 5, 6], [4, 5, 6], [4, 5, 6]])
print(np.mean(a, axis=1)) # [2. 5. 5. 5. 5.]

X_train is [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]] 
then, X_mean = [1457.33, 3.33, 1.33, 40] and sigma = [519.92, 1.53, 0.47, 3.06]
we can do X_train - X_mean because of broadcasting. 
What happens is that X_mean is subtracted from each row of X_train.
Then we divide by sigma. Again, because of broadcasting, each row of X_train is divided by sigma.

After feature scaling, X_norm will be:
[[ 0.69  0.93 -0.79  0.52]
 [-0.87 -0.87  0.4  -0.87]
 [ 0.17 -1.33 -0.79 -0.87]]
"""
# feature scaling by z-score:
X1 = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)



""" feature scaling by min-max scaling (i.e., dividing by the range)

subtract each column by its minimum value and then divide the column by the range of the column. 
For this, find the min and max of each column.

"""
# feature scaling by min-max scaling:
X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

X2 = (X_train - X_min) / (X_max - X_min)



""" feature scaling by mean normalization

this is basically, (x_j - mu) / range
"""
# feature scaling by mean normalization:
X_mean = np.mean(X_train, axis=0)
X_range = np.max(X_train, axis=0) - np.min(X_train, axis=0)

X3 = (X_train - X_mean) / X_range









