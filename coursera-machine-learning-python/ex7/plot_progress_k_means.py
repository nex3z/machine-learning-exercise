import matplotlib.pyplot as plt

from plot_data_points import plot_data_points
from draw_line import draw_line


def plot_progress_k_means(X, history_centroids, idx, K, i):
    """
    Helper function that displays the progress of k-Means as it is running. It is intended for use only with 2D data.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    history_centroids : ndarray, shape (n_max_iters, K, n_features)
        The history of centroids assignment.
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments.
    K : int
        The number of centroids.
    i : int
        Current iteration count.
    """
    plot_data_points(X, idx, K)
    plt.plot(history_centroids[0:i+1, :, 0], history_centroids[0:i+1, :, 1],
             linestyle='', marker='x', markersize=10, linewidth=3, color='k')
    plt.title('Iteration number {}'.format(i + 1))
    for centroid_idx in range(history_centroids.shape[1]):
        for iter_idx in range(i):
            draw_line(history_centroids[iter_idx, centroid_idx, :], history_centroids[iter_idx + 1, centroid_idx, :])
