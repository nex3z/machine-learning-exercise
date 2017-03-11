# Exercise 7-2: Principle Component Analysis and K-Means Clustering
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from feature_normalize import feature_normalize
from pca import pca
from draw_line import draw_line
from project_data import project_data
from recover_data import recover_data
from display_data import display_data
from run_k_means import run_k_means
from k_means_init_centroids import k_means_init_centroids
from plot_data_points import plot_data_points


# ================== Part 1: Load Example Dataset  ===================
print 'Visualizing example dataset for PCA.'

# Load from ex6data1
mat_data = sio.loadmat('ex7data1.mat')
X = mat_data['X']

plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.xlim(0.5, 6.5)
plt.ylim(2, 8)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


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


# =================== Part 3: Dimension Reduction ===================
print 'Dimension reduction on example dataset.'

# Plot the normalized dataset (returned from pca)
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b')
plt.xlim(-4, 3)
plt.ylim(-4, 3)
plt.gca().set_aspect('equal', adjustable='box')

# Project the data onto K = 1 dimension
K = 1
Z = project_data(X_norm, U, K)
print 'Projection of the first example: ', Z[0, ]
print '(this value should be about 1.48127391)'

X_rec = recover_data(Z, U, K)
print 'Approximation of the first example:', X_rec[0, ]
print '(this value should be about  -1.04741883 -1.04741883)'

# Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    draw_line(X_norm[i,:], X_rec[i,:], dash=True)
axes = plt.gca()
axes.set_xlim([-4, 3])
axes.set_ylim([-4, 3])
axes.set_aspect('equal', adjustable='box')
plt.show()


# =============== Part 4: Loading and Visualizing Face Data =============
print 'Loading face dataset.'

# Load Face dataset
mat_data = sio.loadmat('ex7faces.mat')
X = mat_data['X']

plt.figure()
display_data(X[0:100, :])
plt.show()


# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
print 'Running PCA on face dataset.'

# Before running PCA, it is important to first normalize X by subtracting the mean value from each feature
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# Visualize the top 36 eigenvectors found
plt.figure()
display_data(U[:, 0:36].T)
plt.show()


# ============= Part 6: Dimension Reduction for Faces =================
print 'Dimension reduction for face dataset.'
K = 100
Z = project_data(X_norm, U, K)

print 'The projected data Z has a size of:', Z.shape


# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
print 'Visualizing the projected (reduced dimension) faces.'

K = 100
X_rec = recover_data(Z, U, K)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
display_data(X_norm[1:100,:], axes=ax1)
ax1.set_title('Original faces')
ax2 = fig.add_subplot(1, 2, 2)
display_data(X_rec[1:100,:], axes=ax2)
ax2.set_title('Recovered faces')
plt.show()


# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  Reload the image from the previous exercise and run K-Means on it
A = imread('bird_small.png')

A = A.astype(float)/255
img_size = A.shape
X = A.reshape([img_size[0] * img_size[1], img_size[2]])
K = 16
max_iters = 10
initial_centroids = k_means_init_centroids(X, K)
[centroids, idx] = run_k_means(X, initial_centroids, max_iters)

sel = np.random.randint(X.shape[0], size=1000)

# Setup Color Palette
color = cm.rainbow(np.linspace(0, 1, K))

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=color[idx[sel].astype(int), :], marker='o')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()


# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization
# Subtract the mean to use PCA
X_norm, mu, sigma = feature_normalize(X)

# PCA and project the data to 2D
U, S, V = pca(X_norm)
Z = project_data(X_norm, U, 2)

# Plot in 2D
plt.figure()
plot_data_points(Z[sel, ], idx[sel, ], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
