import numpy as np


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    centroids : ndarray, shape (K, n_features)
        The current centroids, where K is the number of centroids.

    Returns
    -------
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments. idx[i] contains the index of the centroid closest to sample i.
    """
    m = X.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        dist = np.sum(np.square(centroids - X[i, :]), axis=1)
        idx[i] = np.argmin(dist)

    return idx
