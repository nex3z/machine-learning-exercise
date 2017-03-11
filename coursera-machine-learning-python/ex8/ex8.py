# Exercise 8-1: Anomaly Detection and Collaborative Filtering
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from estimate_gaussian import estimate_gaussian
from multivariate_gaussian import multivariate_gaussian
from visualize_fit import visualize_fit
from select_threshold import select_threshold


# ================== Part 1: Load Example Dataset  ===================
print 'Visualizing example dataset for outlier detection.'
mat_data = sio.loadmat('ex8data1.mat')
X = mat_data['X']
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

plt.figure()
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== Part 2: Estimate the dataset statistics ===================
print 'Visualizing Gaussian fit.'

# Estimate mu and sigma2
mu, sigma2 = estimate_gaussian(X)

# Returns the density of the multivariate normal at each data point (row) of X
p = multivariate_gaussian(X, mu, sigma2)

# Visualize the fit
plt.figure()
visualize_fit(X, mu, sigma2)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== Part 3: Find Outliers ===================
p_val = multivariate_gaussian(X_val, mu, sigma2)

epsilon, F1 = select_threshold(y_val, p_val)

print 'Best epsilon found using cross-validation:', epsilon
print 'Best F1 on Cross Validation Set:', F1
print '(you should see a value epsilon of about 8.99e-05)'

outliers = np.nonzero(p < epsilon)

plt.figure()
visualize_fit(X, mu, sigma2)
plt.scatter(X[outliers, 0], X[outliers, 1], facecolors='none', edgecolors='r', s=100)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()


# ================== Part 4: Multidimensional Outliers ===================
# Loads the second dataset.
mat_data = sio.loadmat('ex8data2.mat')
X = mat_data['X']
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

# Apply the same steps to the larger dataset
mu, sigma2 = estimate_gaussian(X)

# Training set
p = multivariate_gaussian(X, mu, sigma2)

# Cross-validation set
p_val = multivariate_gaussian(X_val, mu, sigma2)

# Find the best threshold
epsilon, F1 = select_threshold(y_val, p_val)

print 'Best epsilon found using cross-validation:', epsilon
print 'Best F1 on Cross Validation Set:', F1
print '# Outliers found:', np.sum(p < epsilon)
print '(you should see a value epsilon of about 1.38e-18)'
