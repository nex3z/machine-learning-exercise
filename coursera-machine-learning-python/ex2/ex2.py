# Machine Learning Online Class - Exercise 2: Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from sigmoid import sigmoid
from cost_function import cost_function
from cost import cost
from gradient import gradient
from plot_decision_boundary import plot_decision_boundary
from predict import predict

# Load Data
# The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt(open("ex2data1.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]

# ==================== Part 1: Plotting ====================
# We start the exercise by first plotting the data to understand the the problem we are working with.
print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'

plot_data(x, y, ['Admitted', 'Not admitted'])


# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
(m, n) = x.shape

# Add intercept term to x
x = np.concatenate((np.ones((m, 1)), x), axis=1)

# Initialize fitting parameters
theta = np.zeros((n + 1, 1))  # Initialize fitting parameters

# Compute and display initial cost and gradient
j, g = cost_function(theta, x, y)

print 'Cost at initial theta (zeros):', j
print 'Gradient at initial theta (zeros):', g


# ============= Part 3: Optimizing using fmin_tnc  =============
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
theta = result[0].T
print 'Cost at theta found by fminunc:',  cost(theta, x, y)
print 'theta:', theta

plot_decision_boundary(result[0], x, y)


# ============== Part 4: Predict and Accuracies ==============
# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prod = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print 'For a student with scores 45 and 85, we predict an admission probability of', prod

# Compute accuracy on our training set
p = predict(theta, x)

print 'Train Accuracy:', np.mean(p == (y == 1).flatten()) * 100

plt.show()