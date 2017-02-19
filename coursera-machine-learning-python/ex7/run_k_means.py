import numpy as np
import matplotlib.pyplot as plt

from plot_progress_k_means import plot_progress_k_means
from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids


def run_k_means(X, initial_centroids, max_iters, plot_progress=False):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(X.shape[0])

    plt.figure()

    for i in range(max_iters):
        print 'K-Means iteration {}/{}'.format(i + 1, max_iters)

        idx = find_closest_centroids(X, centroids)

        if plot_progress:
            plot_progress_k_means(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        centroids = compute_centroids(X, idx, K)

    plt.show()

    return centroids, idx
