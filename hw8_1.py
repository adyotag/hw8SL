import numpy as np

def sigmoid(x):
    return (1. + np.exp(-x))**-1.

#########################

ITERATIONS = 10
LEARNING_RATE = .50
X = np.asarray([0,1])
y = 1.
W = np.asarray([[.1, .3],[.3, .4]])
b = np.asarray([0.4, 0.6])

for _ in range(ITERATIONS):
    # Get predicted output
    z = sigmoid(np.dot(W, X))
    y_hat = np.dot(z, b)

    # Compute mean-squared error
    error = (y - y_hat)**2

    # compute derivatives of error 
    derivative_b = -2.*(y-y_hat)
    derivative_w = -2.*(y-y_hat)* np.dot(b, z*(1.-z))
    # z*(1-z) is derivative of sigmoid function evaluated at X

    # Update weights
    W -= LEARNING_RATE * derivative_w
    b -= LEARNING_RATE * derivative_b


print('Final Error = ', error)
print('Prediction = ', y_hat)
print('W = ', W)
print('b = ', b)


#
