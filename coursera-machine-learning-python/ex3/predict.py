import numpy as np

from sigmoid import sigmoid


def predict(Theta_1, Theta_2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    Theta_1 : ndarray
        Trained weights of layer 1 of the neural network.
    Theta_2 : ndarray
        Trained weights of layer 2 of the neural network.
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    p : ndarray, shape (n_samples,)
         The prediction for x.
    """

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    A_1 = sigmoid(X.dot(Theta_1.T))
    A_1 = np.hstack((np.ones((m, 1)), A_1))
    A_2 = sigmoid(A_1.dot(Theta_2.T))

    p = np.argmax(A_2, axis=1)
    p += 1  # The theta_1 and theta_2 are loaded from Matlab data, in which the matrix index starts from 1.

    return p
