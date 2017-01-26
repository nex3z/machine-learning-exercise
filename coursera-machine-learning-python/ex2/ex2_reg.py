# Machine Learning Online Class - Exercise 2: Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from map_feature import map_feature
from cost_function_reg import cost_function_reg
from cost_reg import cost_reg
from gradient_reg import gradient_reg
from predict import predict
from plot_decision_boundary import plot_decision_boundary

# Load Data

data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]

plot_data(x, y, ['y = 1', 'y = 0'])
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# plt.show()

# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features
# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
# print x
x = map_feature(x[:, 0:1], x[:, 1:2])

# Initialize fitting parameters
m, n = x.shape
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
lam = 1

j, g = cost_function_reg(initial_theta, x, y, lam)

print 'Cost at initial theta (zeros):', j

# ============= Part 2: Regularization and Accuracies =============

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1 (you should vary this)
lam = 1

# Optimize
result = opt.fmin_tnc(func=cost_reg, x0=initial_theta, fprime=gradient_reg, args=(x, y, lam))
theta = result[0].T

# Plot Boundary
plot_decision_boundary(theta, x, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

# Compute accuracy on our training set
p = predict(theta, x)

print 'Train Accuracy:', np.mean(p == (y == 1).flatten()) * 100


