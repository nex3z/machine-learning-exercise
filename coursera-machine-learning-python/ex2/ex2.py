# Machine Learning Online Class - Exercise 2: Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from cost_function import cost_function
from cost_function_for_opt import cost_opt
from cost_function_for_opt import gradient_opt
from plot_decision_boundary import plot_decision_boundary
from sigmoid import sigmoid
from predict import predict

# Load Data
# The first two columns contains the exam scores and the third column contains the label.
data = np.loadtxt(open("ex2data1.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]


# ==================== Part 1: Plotting ====================
print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'

plot_data(x, y)

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'], loc='upper right', numpoints=1)


# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = x.shape

# Add intercept term to x and X_test
x = np.hstack((np.ones((m, 1)), x))

# Initialize fitting parameters
theta = np.zeros((n + 1, 1))  # Initialize fitting parameters

cost, grad = cost_function(theta, x, y)

print 'Cost at initial theta (zeros):', cost
print 'Gradient at initial theta (zeros):', grad


# ============= Part 3: Optimizing using fmin_tnc  =============
result = opt.fmin_tnc(func=cost_opt, x0=theta, fprime=gradient_opt, args=(x, y), messages=0)
theta = result[0].T
print 'Cost at theta found by fminunc:',  cost_opt(theta, x, y)
print 'theta:', theta

plot_decision_boundary(result[0], x, y)


# ============== Part 4: Predict and Accuracies ==============
# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print 'For a student with scores 45 and 85, we predict an admission probability of', prob

# Compute accuracy on our training set
p = predict(theta, x)

print 'Train Accuracy:', np.mean(p == y) * 100

plt.show()
