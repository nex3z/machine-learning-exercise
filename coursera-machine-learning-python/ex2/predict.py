import numpy as np

from sigmoid import sigmoid


def predict(theta, x):
    p = sigmoid(np.dot(x, theta)) >= 0.5
    return p
