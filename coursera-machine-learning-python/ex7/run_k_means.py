import numpy as np
import matplotlib.pyplot as plt

from plot_progress_k_means import plot_progress_k_means
from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids


def run_k_means(X, initial_centroids, max_iters, plot_progress=False):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    history_centroids = np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
    print history_centroids.shape
    idx = np.zeros(X.shape[0])

    for i in range(max_iters):
        print 'K-Means iteration {}/{}'.format(i + 1, max_iters)
        history_centroids[i, :] = centroids

        idx = find_closest_centroids(X, centroids)

        if plot_progress:
            plt.figure()
            plot_progress_k_means(X, history_centroids, idx, K, i)
            plt.show()

        centroids = compute_centroids(X, idx, K)

    return centroids, idx
