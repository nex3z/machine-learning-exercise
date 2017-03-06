import numpy as np
import matplotlib.pyplot as plt

from plot_progress_k_means import plot_progress_k_means
from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids


def run_k_means(X, initial_centroids, max_iters, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    initial_centroids : ndarray, shape (K, n_features)
        The initial centroids.
    max_iters : int
        Total number of iteration for K-Means to execute.
    plot_progress : bool
        True to plot progress for each iteration.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
        The final centroids.
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments. idx[i] contains the index of the centroid closest to sample i.
    """
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    history_centroids = np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
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
