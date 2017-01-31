import numpy as np

from sigmoid import sigmoid


def predict(theta_1, theta_2, x):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta_1 : ndarray
        Trained weights of layer 1 of the neural network.
    theta_2 : ndarray
        Trained weights of layer 2 of the neural network.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.

    Returns
    -------
    p : ndarray
         The prediction for x, which is an m by 1 array.
    """

    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    a_1 = sigmoid(x.dot(theta_1.T))
    a_1 = np.hstack((np.ones((m, 1)), a_1))
    a_2 = sigmoid(a_1.dot(theta_2.T))

    p = np.argmax(a_2, axis=1)
    p += 1  # The theta_1 and theta_2 are loaded from Matlab data, in which the matrix index starts from 1.

    return p.reshape(m, 1)
