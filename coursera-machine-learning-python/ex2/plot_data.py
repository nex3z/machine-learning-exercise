import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y, labels, show=False):
    """
    Plots the data points x and y into a new figure
    :param x: data on x axis
    :param y: data on y axis
    :param show: whether to show the plot
    """
    plt.figure()

    pos = np.argwhere(y == 1)[:, 0]
    neg = np.argwhere(y == 0)[:, 0]

    plt.plot(x[pos, 0], x[pos, 1], linestyle='', marker='+', color='k', label=labels[0])
    plt.plot(x[neg, 0], x[neg, 1], linestyle='', marker='o', color='y', label=labels[1])

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.legend(loc='upper right', numpoints=1)

    if show:
        plt.show()
