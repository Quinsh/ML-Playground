
"""
here I implement Regularized Logistic Regression.

Regularized logistic regression basically adds a regularization term to the cost function to penalize big parameter values.
With low parameter values, the model is less likely to overfit the training data since the fuction is less likely to be too complex.

"""

""" To Remember

1. The cost function for regularized logistic regression is:
    - J(w, b) = 1/m * Σ(Loss(f(x), y)) + λ/2m * Σ(w^2)
    - where λ is the regularization parameter.
    
2. The gradient of the cost function is:
    - ∂J(w, b)/∂w = 1/m * Σ((ŷ - y)x) + λ/m * w
    - ∂J(w, b)/∂b = 1/m * Σ(ŷ - y)

    λ/m * w is the regularization term. It is likely to be a small number since λ is usually small.
    Thus, each iteration, the weights are decremented by a small number regardless of the cost.
    
""" 

# Regulatized Logistic Regression implementation

# _______________________________ START _______________________________













# _________________________________END__________________________________

