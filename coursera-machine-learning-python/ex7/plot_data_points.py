import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data_points(X, idx, K):
    color = cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(X[:, 0], X[:, 1], c=color[idx.astype(int), :])
