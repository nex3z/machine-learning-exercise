import numpy as np


def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, l):
    """
    Returns the cost and gradient for the collaborative filtering problem.

    Parameters
    ----------
    params : ndarray, shape ((num_users + num_movies) * num_features,)
        Parameters for collaborative filtering.
    Y : ndarray, shape (num_movies, num_users)
        Ratings, where n_movies is the number of movies and n_users is the number of users.
    R : ndarray, shape (num_movies, num_users)
        R(i,j) = 1 if and only if user j gave a rating to movie i.
    num_users : int
        Number of users.
    num_movies : int
        Number of movies.
    num_features : int
        Number of features.
    l : float, shape ((num_users + num_movies) * num_features,)
        Regularization parameter.

    Returns
    -------
    J : numpy.float64
        The cost for collaborative filtering.
    grad : ndarray, shape ((num_users + num_movies) * num_features,)
        The gradient for collaborative filtering.
    """
    X = params[0:num_movies*num_features].reshape((num_movies, num_features))
    Theta = params[num_movies*num_features: ].reshape((num_users, num_features))

    J = 0.5 * np.sum(np.sum(R * np.square(X.dot(Theta.T) - Y)))
    X_grad = (R * (X.dot(Theta.T) - Y)).dot(Theta)
    Theta_grad = (R * (X.dot(Theta.T) - Y)).T.dot(X)

    J = J + 0.5 * l * np.sum(np.square(Theta)) + 0.5 * l * np.sum(np.square(X))
    X_grad = X_grad + l * X
    Theta_grad = Theta_grad + l * Theta

    grad = np.hstack((X_grad.ravel(), Theta_grad.ravel()))

    return J, grad
