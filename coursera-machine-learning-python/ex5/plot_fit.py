import numpy as np
import matplotlib.pyplot as plt

from poly_features import poly_features
from feature_normalize import feature_normalize


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.

    Parameters
    ----------
    min_x : float
        Minimum value of features.
    max_x : float
        Maximum value of features.
    mu : float
        Mean value of features.
    sigma : float
        Standard deviation of features.
    theta : ndarray
        Linear regression parameter.
    p : int
        Power of polynomial fit.
    """
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    x_poly = poly_features(x, p)
    x_poly, dummy_mu, dummy_sigma = feature_normalize(x_poly, mu, sigma)
    x_poly = np.hstack((np.ones((x_poly.shape[0], 1)), x_poly))
    plt.plot(x, x_poly.dot(theta), linestyle='--', marker='', color='b')
