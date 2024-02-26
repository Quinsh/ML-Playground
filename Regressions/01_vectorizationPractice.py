import numpy as np    # it is an unofficial standard to use np for numpy
import time

## __________________________________
## vectorization practice with NumPy
## __________________________________

# Creating array
a = np.zeros((5, ))
a = np.random.random_sample(4)
a = np.random.rand(4)
a = np.arange(10)

a = np.array([1, 2.5, 3]) # manually create

# ___________________________________
# Operation on Vectors
b = np.arange(10)

# see shape
print(b.shape)
print(b[1].shape)
# negate elements on b
c = -b
# sum and mean
S = np.sum(c)
M = np.mean(c)
# power
d = c ** 2
# addition and subtraction
e = b-c
# scalar multiplication
e = 2 * b
# dot product
dot = np.dot(b, e)

# ___________________________________
# Operation on Matrices

# reshape 
m1 = np.zeros(20)
m1 = m1.reshape(-1, 5)
# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")
# access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

