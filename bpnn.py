import numpy as np

# NN Size: 4 x 10 x 6 x 2

# Inputs
x = np.array([[0, 0, 1, 1],
              [0, 1, 1, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 1]])

# Output
y = np.array([[0, 1], [0.5, 0], [1, 0.5], [0, 1]])
np.random.seed(1)

size0 = 10  # Number of first hide layer neurons
size1 = 6  # Number of second hide layer neurons

# Layers( weights, bias)
W0 = 2 * np.random.random((4, size0)) - 1
b0 = 0.1 * np.ones((size0,))
W1 = 2 * np.random.random((size0, size1)) - 1
b1 = 0.1 * np.ones((size1,))
W2 = 2 * np.random.random((size1, 2)) - 1
b2 = 0.1 * np.ones((2,))


# Nonlinear functions
def sigmold(X, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-X))
    else:
        return X * (1 - X)


def softplus(X, derive=False):
    if not derive:
        return np.log(1 + np.exp(X))
    else:
        return 1 / (1 + np.exp(-X))


def relu(X, derive=False):
    if not derive:
        return np.maximum(0, X)
    else:
        return (X > 0).astype(float)


def tanh(X, derive=False):
    if not derive:
        return np.tanh(X)
    else:
        return 1.0 - X ** 2


nonline = tanh
rate = 0.1
# Trainling
training_times = 6000
for i in range(training_times):
    # Layer1
    A0 = np.dot(x, W0) + b0
    Z0 = nonline(A0)

    # Layer2
    A1 = np.dot(Z0, W1) + b1
    Z1 = nonline(A1)

    # Layer3
    A2 = np.dot(Z1, W2) + b2
    _y = Z2 = nonline(A2)
    cost = _y - y
    # print("Cost x:{}".format(np.mean(np.abs(cost))))

    # Calc deltas
    delta_A2 = cost * nonline(Z2, derive=True)
    delta_b2 = delta_A2.sum(axis=0)
    delta_W2 = np.dot(Z1.T, delta_A2)

    delta_A1 = np.dot(delta_A2, W2.T) * nonline(Z1, derive=True)
    delta_b1 = delta_A1.sum(axis=0)
    delta_W1 = np.dot(Z0.T, delta_A1)

    delta_A0 = np.dot(delta_A1, W1.T) * nonline(Z0, derive=True)
    delta_b0 = delta_A0.sum(axis=0)
    delta_W0 = np.dot(x.T, delta_A0)

    # Apply deltas
    W2 -= rate * delta_W2
    b2 -= rate * delta_b2
    W1 -= rate * delta_W1
    b1 -= rate * delta_b1
    W0 -= rate * delta_W0
    b0 -= rate * delta_b0

else:
    # Print cost, weights, bias, last output
    print("Cost:{}".format(np.mean(np.abs(cost))))
    print(np.around(W0, 3), "W0")
    print(np.around(b0, 3), "b0")
    print(np.around(W1, 3), "W1")
    print(np.around(b1, 3), "b1")
    print(np.around(W2, 3), "W2")
    print(np.around(b2, 3), "b2")
    print(np.around(_y, 2), "_y")
