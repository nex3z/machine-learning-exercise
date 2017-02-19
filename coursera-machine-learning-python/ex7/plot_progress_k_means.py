import numpy as np
import matplotlib.pyplot as plt

from plot_data_points import plot_data_points


def plot_progress_k_means(X, centroids, previous, idx, K, i):
    plot_data_points(X, idx, K)
    plt.plot(centroids[:, 0], centroids[:, 1], linestyle='', marker='x', markersize=10, linewidth=3, color='k')
    plt.title('Iteration number {}'.format(i + 1))
    for i in range(centroids.shape[0]):
        draw_line(centroids[i, :], previous[i, :])


def draw_line(p1, p2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k')
