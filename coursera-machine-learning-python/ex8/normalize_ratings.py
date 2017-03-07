import numpy as np


def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).

    Parameters
    ----------
    Y : ndarray, shape (n_movies, n_users)
        Ratings, where n_movies is the number of movies and n_users is the number of users.
    R : ndarray, shape (n_samples, n_users)
        R(i,j) = 1 if and only if user j gave a rating to movie i.

    Returns
    -------
    Y_norm : ndarray, shape (n_movies, n_users)
        Normalized Y with each movie has am average rating of 0.
    Y_mean : ndarray, shape (n_movies,)
        Mean rating for each movie.
    """
    m = Y.shape[0]
    Y_mean = np.zeros(m)
    Y_norm = np.zeros(Y.shape)

    for i in range(m):
        idx = np.nonzero(R[i, ] == 1)
        Y_mean[i] = np.mean(Y[i, idx])
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

    return Y_norm, Y_mean