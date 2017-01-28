import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y, show=False):
    """
    Plots the data points x and y into a new figure.

    Parameters
    ----------
    x : array
        Data on x axis. It's a m by 2 matrix, where m is the number of data samples and the number of features is 2.
    y : array
        Data on y axis, m by 1 matrix.
    show : bool
        True to show the plot immediately.
    """
    plt.figure()

    pos = np.argwhere(y.flatten() == 1)
    neg = np.argwhere(y.flatten() == 0)

    plt.plot(x[pos, 0], x[pos, 1], linestyle='', marker='+', color='k')
    plt.plot(x[neg, 0], x[neg, 1], linestyle='', marker='o', color='y')

    if show:
        plt.show()
