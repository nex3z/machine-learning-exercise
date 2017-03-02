import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    """
    Plots the data points x and y into a new figure.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Data on x axis.
    y : array-like, shape (n_samples,)
        Data on y axis.
    """
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)

    plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
    plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')
