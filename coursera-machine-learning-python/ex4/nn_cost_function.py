import numpy as np

from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, l):
    """
    Implements the neural network cost function for a two layer neural network which performs classification.

    Parameters
    ----------
    nn_params : ndarray
        Parameters for the neural network, "unrolled" into a vector.
    input_layer_size : int
        The size of the input layer.
    hidden_layer_size : int
        The size of the hidden layer.
    num_labels : int
        The number of labels.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    l : float
        Regularization parameter.

    Returns
    -------
    J : numpy.float64
        The cost of the neural network w.r.t. the parameters.
    grad: ndarray
        The gradient of the neural network w.r.t. the parameters.
    """
    theta_1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1)), ],
                         (hidden_layer_size, input_layer_size + 1), order='F')
    theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):, ],
                         (num_labels, hidden_layer_size + 1), order='F')

    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))

    z_2 = x.dot(theta_1.T)
    a_2 = sigmoid(z_2)
    a_2 = np.hstack((np.ones((m, 1)), a_2))

    z_3 = a_2.dot(theta_2.T)
    a_3 = sigmoid(z_3)

    y_mat = np.zeros((m, num_labels))
    for i in range(m):
        y_mat[i, y[i] - 1] = 1

    j = 0
    for i in range(m):
        j += np.log(a_3[i, ]).dot(-y_mat[i, ].T) - np.log(1 - a_3[i, ]).dot(1 - y_mat[i, ].T)
    j /= m

    theta_1_square = np.square(theta_1[:, 1:])
    theta_2_square = np.square(theta_2[:, 1:])
    reg = 1.0 * l / (2 * m) * (np.sum(theta_1_square) + np.sum(theta_2_square))
    j += reg

    d_3 = a_3 - y_mat
    D_2 = d_3.T.dot(a_2)

    z_2 = np.hstack((np.ones((m, 1)), z_2))
    d_2 = d_3.dot(theta_2) * sigmoid_gradient(z_2)
    d_2 = d_2[:, 1:]
    D_1 = d_2.T.dot(x)

    theta_1_grad = 1.0 * D_1 / m
    theta_1_grad[:, 1:] = theta_1_grad[:, 1:] + 1.0 * l / m * theta_1[:, 1:]

    theta_2_grad = 1.0 * D_2 / m
    theta_2_grad[:, 1:] = theta_2_grad[:, 1:] + 1.0 * l / m * theta_2[:, 1:]

    grad = np.hstack((theta_1_grad.flatten('F'), theta_2_grad.flatten('F')))
    grad = grad.reshape(len(grad), 1)

    return j, grad
