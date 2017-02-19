import numpy as np
import matplotlib.pyplot as plt

from plot_data_points import plot_data_points


def plot_progress_k_means(X, history_centroids, idx, K, i):
    plot_data_points(X, idx, K)
    plt.plot(history_centroids[0:i+1, :, 0], history_centroids[0:i+1, :, 1],
             linestyle='', marker='x', markersize=10, linewidth=3, color='k')
    plt.title('Iteration number {}'.format(i + 1))
    print history_centroids.shape
    for centroid_idx in range(history_centroids.shape[1]):
        for iter_idx in range(i):
            draw_line(history_centroids[iter_idx, centroid_idx, :], history_centroids[iter_idx + 1, centroid_idx, :])


def draw_line(p1, p2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k')
