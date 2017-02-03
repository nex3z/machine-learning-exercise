import numpy as np
import matplotlib.pyplot as plt

from map_feature import map_feature


def plot_decision_boundary(theta, x, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    """
    if x.shape[1] <= 3:
        plot_x = np.array([np.amin(x[:, 1]) - 2, np.amax(x[:, 1]) + 2])
        plot_y = -1.0 / theta[2] * (theta[1] * plot_x + theta[0])
        print plot_x, plot_y
        plt.plot(plot_x, plot_y)
    else:
        u = np.linspace(-1, 1.5, 50)
        u.resize((len(u), 1))
        v = np.linspace(-1, 1.5, 50)
        v.resize((len(v), 1))
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = map_feature(u[i, 0:1].reshape((1, 1)), v[j, 0:1].reshape((1, 1))).dot(theta)
        z = z.T
        u, v = np.meshgrid(u, v)
        cs = plt.contour(u, v, z, levels=[0])
        fmt = {}
        strs = ['Decision boundary']
        for l, s in zip(cs.levels, strs):
            fmt[l] = s

        plt.clabel(cs, cs.levels[::2], inline=True, fmt=fmt, fontsize=10)
