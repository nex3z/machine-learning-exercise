# Exercise 7 | Principle Component Analysis and K-Means Clustering
import numpy as np
import scipy.io as sio

from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids
from run_k_means import run_k_means


# ================= Part 1: Find Closest Centroids ====================
print 'Finding closest centroids...'

# Load an example dataset that we will be using
mat_data = sio.loadmat('ex7data2.mat')
X = mat_data['X']

# Select an initial set of centroids
K = 3  # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
print initial_centroids

# Find the closest centroids for the examples using the initial_centroids
idx = find_closest_centroids(X, initial_centroids)

print 'Closest centroids for the first 3 examples: '
print idx[0:3]
print '(the closest centroids should be 0, 2, 1 respectively)'


# ===================== Part 2: Compute Means =========================
print 'Computing centroids means...'

# Compute means based on the closest centroids found in the previous part.
centroids = compute_centroids(X, idx, K)

print 'Centroids computed after initial finding of closest centroids:'
print centroids
print '(the centroids should be \n[ 2.428301 3.157924 ]\n[ 5.813503 2.633656 ]\n[ 7.119387 3.616684 ]'


# =================== Part 3: K-Means Clustering ======================
print 'Running K-Means clustering on example dataset...'

# Load an example dataset
mat_data = sio.loadmat('ex7data2.mat')
X = mat_data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values but in practice you want to generate them automatically,
# such as by settings them to be random examples (as can be seen in k_means_init_centroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm
centroids, idx = run_k_means(X, initial_centroids, max_iters, True)
print 'K-Means Done.'

