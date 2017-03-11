# Exercise 1-2: Linear regression with multiple variables
import numpy as np
import matplotlib.pyplot as plt

from feature_normalize import feature_normalize
from gradient_descent_multi import gradient_descent_multi
from normal_eqn import normal_eqn

# ================ Part 1: Feature Normalization ================
print 'Loading data...'
# Load Data
data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Print out some data points
print 'First 10 examples from the dataset:\n',
for i in range(10):
    print 'x =', X[i, ], ', y =', y[i]

# Scale features and set them to zero mean
print 'Normalizing Features...'
X, mu, sigma = feature_normalize(X)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))


# ================ Part 2: Gradient Descent ================
print 'Running gradient descent...'

# Choose some alpha value
alpha = 0.15
num_iters = 400

# Init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(range(1, num_iters + 1), J_history, color='b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print 'Theta computed from gradient descent: '
print theta

#  Estimate the price of a 1650 sq-ft, 3 br house
normalize_test_data = ((np.array([1650, 3]) - mu) / sigma)
normalize_test_data = np.hstack((np.ones(1), normalize_test_data))
price = normalize_test_data.dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house:', price


# ================ Part 3: Normal Equations ================
print 'Solving with normal equations...'

data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))

# Calculate the parameters from the normal equation
theta = normal_eqn(X, y)
print 'Theta computed from the normal equations: '
print theta

price = np.array([1, 1650, 3]).dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price
