# Machine Learning Online Class - Exercise 1: Linear regression with multiple variables
import numpy as np
import matplotlib.pyplot as plt

from feature_normalize import feature_normalize
from gradient_descent_multi import gradient_descent_multi
from normal_eqn import normal_eqn

# ================ Part 1: Feature Normalization ================
print 'Loading data...'
# Load Data
data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

# Print out some data points
print 'First 10 examples from the dataset:\n',
for i in range(10):
    print 'x =', x[i, ], ', y =', y[i, ]

# Scale features and set them to zero mean
print 'Normalizing Features...'

x, mu, sigma = feature_normalize(x)

# Add intercept term to X
x = np.concatenate((np.ones((m, 1)), x), axis=1)


# ================ Part 2: Gradient Descent ================
print 'Running gradient descent...'

# Choose some alpha value
alpha = 0.15
num_iters = 400

# Init theta and run gradient descent
theta = np.zeros((3, 1))
theta, j_history = gradient_descent_multi(x, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(range(1, num_iters + 1), j_history, color='b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print 'Theta computed from gradient descent: '
print theta

#  Estimate the price of a 1650 sq-ft, 3 br house
normalize_test_data = ((np.array([1650, 3]) - mu) / sigma).reshape(1, 2)
normalize_test_data = np.concatenate((np.ones((1, 1)), normalize_test_data), axis=1)
price = normalize_test_data.dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house:', price


# ================ Part 3: Normal Equations ================
print 'Solving with normal equations...'

data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
x = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

# Add intercept term to X
x = np.concatenate((np.ones((m, 1)), x), axis=1)

# Calculate the parameters from the normal equation
theta = normal_eqn(x, y)
print 'Theta computed from the normal equations: '
print theta

price = np.array([1, 1650, 3]).dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price


# Show the plots
plt.show()
