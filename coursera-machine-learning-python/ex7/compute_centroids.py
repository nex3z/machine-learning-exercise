import numpy as np


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        x = X[idx == k]
        centroids[k, :] = np.mean(x, axis=0)

    return centroids