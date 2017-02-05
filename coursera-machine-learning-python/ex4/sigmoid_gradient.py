from sigmoid import sigmoid


def sigmoid_gradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z

    Parameters
    ----------
    z : array_like
        Variable for sigmoid function.

    Returns
    -------
    ndarray
        The gradient of the sigmoid of each value of z.
    """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g
