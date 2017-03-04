import numpy as np

from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    """
    Implements the neural network cost function for a two layer neural network which performs classification.

    Parameters
    ----------
    nn_params : ndarray, shape (n_params,)
        Parameters for the neural network, "unrolled" into a vector.
    input_layer_size : int
        The size of the input layer.
    hidden_layer_size : int
        The size of the hidden layer.
    num_labels : int
        The number of labels.
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    l : float
        Regularization parameter.

    Returns
    -------
    j : numpy.float64
        The cost of the neural network w.r.t. the parameters.
    grad: ndarray, shape (n_params,)
        The gradient of the neural network w.r.t. the parameters.
    """
    Theta_1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1)), ],
                         (hidden_layer_size, input_layer_size + 1))
    Theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):, ],
                         (num_labels, hidden_layer_size + 1))

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))

    Z_2 = X.dot(Theta_1.T)
    A_2 = sigmoid(Z_2)
    A_2 = np.hstack((np.ones((m, 1)), A_2))

    Z_3 = A_2.dot(Theta_2.T)
    A_3 = sigmoid(Z_3)

    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    j = 0.0
    for i in range(m):
        j += np.log(A_3[i, ]).dot(-Y[i, ].T) - np.log(1 - A_3[i, ]).dot(1 - Y[i, ].T)
    j /= m

    Theta_1_square = np.square(Theta_1[:, 1:])
    Theta_2_square = np.square(Theta_2[:, 1:])
    reg = 1.0 * l / (2 * m) * (np.sum(Theta_1_square) + np.sum(Theta_2_square))
    j += reg

    d_3 = A_3 - Y
    D_2 = d_3.T.dot(A_2)

    Z_2 = np.hstack((np.ones((m, 1)), Z_2))
    d_2 = d_3.dot(Theta_2) * sigmoid_gradient(Z_2)
    d_2 = d_2[:, 1:]
    D_1 = d_2.T.dot(X)

    Theta_1_grad = 1.0 * D_1 / m
    Theta_1_grad[:, 1:] = Theta_1_grad[:, 1:] + 1.0 * l / m * Theta_1[:, 1:]

    Theta_2_grad = 1.0 * D_2 / m
    Theta_2_grad[:, 1:] = Theta_2_grad[:, 1:] + 1.0 * l / m * Theta_2[:, 1:]

    grad = np.hstack((Theta_1_grad.ravel(), Theta_2_grad.ravel()))

    return j, grad
