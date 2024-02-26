

""" To remember

1. np.c_ is used to concatenate the arrays.
2. feature that is linear with the target will automatically be attributed higher weight.
3. normalize the features before running gradient descent!!

"""

# ________________________________________START____________________________________________________

import numpy as np

# suppose we have x = [1, 2, 3, 4, 5] and y = [1, 4, 9, 16, 25]
# we can see that y = x^2

# we need to fit a y = x^2 + b model instead of y = x

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

x = x ** 2
X = x.reshape(-1, 1) # we need to reshape the array to make it a 2D array

# if we run gradient descent now, we fit y = x^2 + b. This is feature engineering.


# But we can also fit some y = w0x^3 + w1x^2 + w2x + b model. The weights are going to adjust automatically,
# so that if x^3 is not important, the weight will be 0.

X = np.c_[x**3, x**2, x] # we can use np.c_ to concatenate the arrays

# if we now run gradient descent, we fit y = w0x^3 + w1x^2 + w2x + b. But w0 and w2 will converge to 0 
# as it perceives that x^3 and x are not important.
# In alternate view, gradient descent picks the best features that are linear relative to the target.
# If x^2 is linear relative to the target, it will pick x^2.

# Also, don't forget to normalize the features:
X = zscore_normalize_features(X) # we need to implement this function