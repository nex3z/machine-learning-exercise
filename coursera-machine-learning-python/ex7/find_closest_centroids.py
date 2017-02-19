import numpy as np


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        dist = np.sum(np.square(centroids - X[i, :]), axis=1)
        idx[i] = np.argmin(dist)

    return idx
