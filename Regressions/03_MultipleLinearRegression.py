

""" What I've learned:

when feature is not scaled correctly, gradients can become huge. So setting up the right alpha value is essential.
If alpha is big, the algorithm will overshoot the minimum.

In this experiment alpha needed to be set to 0.0000005 not to overshoot.

"""

""" To Remember:

Just remember the gradient computing function. That's the hardest part of this.

"""

import copy, math
import numpy as np
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# set train datas and initial parameter values
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

w_init = np.array([1, 10, -30, -30])
b_init = 500

# predicting function
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

# cost function J
def cost_J(X, y, w, b):
    m = X.shape[0] # num of rows of vector x (which is our input)
    J = 0
    for i in range(m):
        J += ((np.dot(w, X[i]) + b) - y[i]) ** 2
    J /= 2 * m
    return J

# compute the gradients (hardest part in this file) 
def compute_gradients(X, y, w, b):
    m, n = X.shape
    
    dJ_dw = np.zeros((n,))
    dJ_db = 0.
    
    for i in range(m):
        error = (np.dot(w, X[i]) + b) - y[i]
        # print(f"error is: {error}")
        for j in range(n):
            dJ_dw[j] = dJ_dw[j] + error * X[i,j]
            # print(f"dJ_dw[j] is: {error * X[i,j]}")
        dJ_db += error
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m

    return dJ_dw, dJ_db

# do the gradient descent
def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, iters):
    """
    Args:
    X (ndarray (m,n))   : Data, m examples with n features
    y (ndarray (m,))    : target values
    w_in (ndarray (n,)) : initial model parameters  
    b_in (scalar)       : initial model parameter
    cost_function       : function to compute cost
    gradient_function   : function to compute the gradient
    alpha (float)       : Learning rate
    num_iters (int)     : number of iterations to run gradient descent
    
    Returns:
    w (ndarray (n,)) : Updated values of parameters 
    b (scalar)       : Updated value of parameter 
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    
    for i in range(iters):
        dJ_dw, dJ_db = gradient_function(X, y, w, b)
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db
        
        J_history.append(cost_J(X, y, w, b))
        
        
    return w, b, J_history


w, b, J_hist = gradient_descent(X_train, y_train, w_init, b_init, compute_gradients, 0.0000005, 100000)

for i, j in enumerate(J_hist):
    if (i % 10000 == 0):
        print(j)
    
print(w, b)
    
print(f"prediction: {predict([2104, 5, 1, 45], w, b)}")
