# Machine Learning Online Class - Exercise 2: Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from map_feature import map_feature
from cost_function_reg import cost_function_reg
from cost_function_reg_for_opt import cost_reg_opt
from cost_function_reg_for_opt import gradient_reg_opt
from plot_decision_boundary import plot_decision_boundary
from predict import predict

# Load Data
# The first two columns contains the X values and the third column contains the label (y).
data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]

plot_data(x, y)
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right', numpoints=1)


# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features
# Note that map_feature also adds a column of ones for us, so the intercept term is handled
x = map_feature(x[:, 0:1], x[:, 1:2])

m, n = x.shape
# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
l = 1.0

j, g = cost_function_reg(initial_theta, x, y, l)

print 'Cost at initial theta (zeros):', j


# ============= Part 2: Regularization and Accuracies =============
# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to t1 (you should vary this)
l = 1.0

# Optimize
result = opt.fmin_tnc(func=cost_reg_opt, x0=initial_theta, fprime=gradient_reg_opt, args=(x, y, l), messages=0)
theta = result[0].T

# Plot Boundary
plot_decision_boundary(theta, x, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Compute accuracy on our training set
p = predict(theta, x)

print 'Train Accuracy:', np.mean(p == y) * 100

plt.show()
