# Exercise 7 | Principle Component Analysis and K-Means Clustering
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from feature_normalize import feature_normalize
from pca import pca
from draw_line import draw_line


# ================== Part 1: Load Example Dataset  ===================
print 'Visualizing example dataset for PCA.'

# Load from ex6data1
mat_data = sio.loadmat('ex7data1.mat')
X = mat_data['X']

plt.figure()
plt.plot(X[:, 0], X[:, 1], marker='o', color='b', linestyle='')
plt.axis('equal')


# =============== Part 2: Principal Component Analysis ===============
print 'Running PCA on example dataset.'

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

plt.figure()
draw_line(mu, mu + 1.5 * S[0] * U[:,0].T)
draw_line(mu, mu + 1.5 * S[1] * U[:,1].T)
plt.show()

print 'Top eigenvector:'
print 'U = ', U[:, 0]
print '(you should expect to see -0.70710678 -0.70710678)'
