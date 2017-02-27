import numpy as np


def rand_initialize_weights(l_in, l_out):
    """
    Randomly initialize the weights of a layer with l_in incoming connections and l_out outgoing connections.

    Parameters
    ----------
    l_in : int
        Number of incoming connections.
    l_out : int
        Number of outgoing connections.

    Returns
    -------
    ndarray
        The randomly initialized weight.
    """
    epsilon_init = 0.12

    # Note that w should be set to a matrix of size(l_out, 1 + l_in) as the first column of W handles the "bias" terms
    W = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init

    return W
