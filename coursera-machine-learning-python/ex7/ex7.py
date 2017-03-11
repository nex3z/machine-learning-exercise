# Exercise 7-1: Principle Component Analysis and K-Means Clustering
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread

from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids
from run_k_means import run_k_means
from k_means_init_centroids import k_means_init_centroids


# ================= Part 1: Find Closest Centroids ====================
print 'Finding closest centroids...'

# Load an example dataset that we will be using
mat_data = sio.loadmat('ex7data2.mat')
X = mat_data['X']

# Select an initial set of centroids
K = 3  # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

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
print '(the centroids should be [ 2.428301 3.157924 ], [ 5.813503 2.633656 ], [ 7.119387 3.616684 ])'


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


# ============= Part 4: K-Means Clustering on Pixels ===============
print 'Running K-Means clustering on pixels from an image.'

# Load an image of a bird
A = imread('bird_small.png')
A = A.astype(float)/255

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
X = A.reshape([img_size[0] * img_size[1], img_size[2]])

K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly.
initial_centroids = k_means_init_centroids(X, K)

# Run K-Means
centroids, idx = run_k_means(X, initial_centroids, max_iters)


# ================= Part 5: Image Compression ======================
print 'Applying K-Means to compress an image.'
idx = find_closest_centroids(X, centroids)

# Recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value.
X_recovered = centroids[idx.astype(int), :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size)

fig = plt.figure()
# Display the original image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(A)
ax1.set_title('Original')
# Display compressed image side by side
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(X_recovered)
ax2.set_title('Compressed, with {} colors.'.format(K))
plt.show()
