

x_train = [1,2,3,4,5,6,7,8]
y_train = [2,5,10,8,9,12,14,15]

# Let linear function be f = wx + b. We need to find w and b that minimizes J (cost function).

# choose initial values for w, b
w = 1
b = 0

# function finding the partial derivative d/dw
def del_w(x_data, y_data, w, b):
    sum = 0
    for x, y in zip (x_data, y_data):
        sum += (x * w + b - y) * x
    sum /= len(x_data)
    return sum

# function finding the partial derivative d/db
def del_b(x_data, y_data, w, b):
    sum = 0
    for x, y in zip (x_data, y_data):
        sum += (x * w + b - y)
    sum /= len(x_data)
    return sum

# Gradient Descent Algorithm
epsilon = 0.00001
alpha = 0.002
cnt = 0

while (True):
    # do a simultaneous update (w shouldn't be updated before computing db. dw and db hence should be calculated at the same time)
    dw = alpha * del_w(x_train, y_train, w, b)
    db = alpha * del_b(x_train, y_train, w, b)
    
    w = w - dw
    b = b - db
    cnt += 1
    
    # just printing the process
    if (cnt % 100 == 0):
        print(f"dw = {dw}, db = {db}")
    
    # how should I know when to stop? What should be less tha epsilon?
    if (abs(dw) + abs(db) < epsilon): break
    
print(f"y = {w}x + {b}")
print(f"Total iteration: {cnt}")

