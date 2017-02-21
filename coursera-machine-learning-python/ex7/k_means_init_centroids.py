import numpy as np


def k_means_init_centroids(X, K):
    rand_idx = np.random.permutation(X.shape[0])
    centroids = X[rand_idx[0:K], :]
    return centroids
