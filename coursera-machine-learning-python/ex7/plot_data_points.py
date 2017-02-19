import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data_points(X, idx, K):
    """
    Plots data points in X, coloring them so that those with the same index assignments in idx have the same color.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments.
    K : int
        The number of centroids.
    """
    color = cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(X[:, 0], X[:, 1], c=color[idx.astype(int), :])
