import numpy as np
import matplotlib.pyplot as plt

from map_feature import map_feature


def plot_decision_boundary(theta, x, y):
    if x.shape[1] <= 3:
        plot_x = np.array([np.amin(x[:, 1]) - 2, np.amax(x[:, 1]) + 2])
        plot_y = -1.0 / theta[2] * (theta[1] * plot_x + theta[0])
        print plot_x, plot_y
        plt.plot(plot_x, plot_y)
    else:
        u = np.linspace(-1, 1.5, 50)
        u = u.reshape((len(u), 1))
        v = np.linspace(-1, 1.5, 50)
        v = v.reshape((len(v), 1))
        z = np.zeros((len(u), len(v)))
        print u.shape, u[1, :].reshape((1,1)).shape
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = map_feature(u[i, 0:1].reshape((1,1)), v[j, 0:1].reshape((1,1))).dot(theta);
        z = z.T
        u, v = np.meshgrid(u, v)
        cs = plt.contour(u, v, z, levels=[0])
        fmt = {}
        strs = ['Decision boundary']
        for l, s in zip(cs.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        plt.clabel(cs, cs.levels[::2], inline=True, fmt=fmt, fontsize=10)
