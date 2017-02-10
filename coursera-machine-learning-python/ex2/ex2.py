# Exercise 2: Logistic Regression
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plot_data import plot_data
from cost_function import cost_function
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
theta, nfeval, rc = opt.fmin_tnc(func=cost_function, x0=theta, args=(x, y), messages=0)
if rc == 0:
    print 'Local minimum reached after', nfeval, 'function evaluations.'

# Resize theta to an (n + 1) by 1 vector. It's not required nor necessary here but I'll just keep theta the same format
# as in the original Matlab code.
theta.resize(n + 1, 1)

# Print theta to screen
cost, _ = cost_function(theta, x, y)
print 'Cost at theta found by fminunc:', cost
print 'theta:', theta

# Plot Boundary
plot_decision_boundary(theta, x, y)


# ============== Part 4: Predict and Accuracies ==============
# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print 'For a student with scores 45 and 85, we predict an admission probability of', prob

# Compute accuracy on our training set
p = predict(theta, x)

print 'Train Accuracy:', np.mean(p == y) * 100

plt.show()
