# Exercise 6 | Support Vector Machines
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

from plot_data import plot_data
from gaussian_kernel import gaussian_kernel
from visualize_boundary_linear import visualize_boundary_linear
from visualize_boundary import visualize_boundary
from data3_params import data3_params


# =============== Part 1: Loading and Visualizing Data ================
print 'Loading and Visualizing Dataset 1...'

# Load from ex6data1
mat_data = sio.loadmat('ex6data1.mat')
X = mat_data['X']
y = mat_data['y']

# Plot training data
plot_data(X, y)
plt.title("Dataset 1")


# ==================== Part 2: Training Linear SVM ====================
# Load from ex6data1
mat_data = sio.loadmat('ex6data1.mat')
X = mat_data['X']
y = mat_data['y']

print 'Training Linear SVM...'
# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
C = 1
clf = svm.LinearSVC(C=C)
clf.fit(X, y.ravel())
print 'score:', clf.score(X, y)

visualize_boundary_linear(X, y, clf)
plt.title("Dataset 1")


# =============== Part 3: Implementing Gaussian Kernel ===============
print 'Evaluating the Gaussian Kernel...'
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print 'Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {}: \n\t{}\n' \
      '(for sigma = 2, this value should be about 0.324652)'.format(sigma, sim)


# =============== Part 4: Visualizing Dataset 2 ================
print 'Loading and Visualizing Dataset 2...'

# Load from ex6data2
mat_data = sio.loadmat('ex6data2.mat')
X = mat_data['X']
y = mat_data['y']

# Plot training data
plot_data(X, y)
plt.title("Dataset 2")


# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
print 'Training SVM with RBF Kernel...'
# Load from ex6data1
mat_data = sio.loadmat('ex6data2.mat')
X = mat_data['X']
y = mat_data['y']

# SVM Parameters
C = 100
gamma = 10

clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
clf.fit(X, y.ravel())
print 'score:', clf.score(X, y)

visualize_boundary(X, y, clf)
plt.title("Dataset 2")


# =============== Part 6: Visualizing Dataset 3 ================
print 'Loading and Visualizing Data 3...'

# Load from ex6data3
mat_data = sio.loadmat('ex6data3.mat')
X = mat_data['X']
y = mat_data['y']

# Plot training data
plot_data(X, y)

# Plot training data
plt.title("Dataset 3")


# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
# Load from ex6data3
mat_data = sio.loadmat('ex6data3.mat')
X = mat_data['X']
y = mat_data['y']
X_val = mat_data['Xval']
y_val = mat_data['yval']

# Try different SVM Parameters here
c, gamma = data3_params(X, y, X_val, y_val)
print c, gamma
# Train the SVM
clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
clf.fit(X, y.ravel())

visualize_boundary(X, y, clf)
plt.title("Dataset 3")

plt.show()
