# Exercise 2-2: Logistic Regression with Regularization
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from map_feature import map_feature
from cost_function_reg import cost_function_reg
from plot_decision_boundary import plot_decision_boundary
from predict import predict


# Load Data
# The first two columns contains the X values and the third column contains the label (y).
data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]

plt.figure()
plot_data(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right', numpoints=1)
plt.show()


# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features
# Note that map_feature also adds a column of ones for us, so the intercept term is handled
X = map_feature(X[:, 0], X[:, 1])
m, n = X.shape

# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to 1
l = 1.0

cost, _ = cost_function_reg(initial_theta, X, y, l)

print 'Cost at initial theta (zeros):', cost


# ============= Part 2: Regularization and Accuracies =============
# Initialize fitting parameters
initial_theta = np.zeros(n)

# Set regularization parameter lambda to t1 (you should vary this)
l = 1.0

# Optimize
theta, nfeval, rc = opt.fmin_tnc(func=cost_function_reg, x0=initial_theta, args=(X, y, l), messages=0)

# Plot Boundary
plt.figure()
plot_data(X[:, 1:], y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right', numpoints=1)

plot_decision_boundary(theta, X, y)
plt.show()

# Compute accuracy on our training set
p = predict(theta, X)

print 'Train Accuracy:', np.mean(p == y) * 100
