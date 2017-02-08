# Exercise 5 | Regularized Linear Regression and Bias-Variance
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from linear_reg_cost_function import linear_reg_cost_function
from train_linear_reg import train_linear_reg
from learning_curve import learning_curve
from poly_features import poly_features
from feature_normalize import feature_normalize
from plot_fit import plot_fit
from validation_curve import validation_curve


# =========== Part 1: Loading and Visualizing Data =============
print 'Loading and Visualizing Data...'
mat_data = sio.loadmat('ex5data1.mat')
x = mat_data['X']
y = mat_data['y']
x_test = mat_data['Xtest']
y_test = mat_data['ytest']
x_val = mat_data['Xval']
y_val = mat_data['yval']
m, n = x.shape
m_val = x_val.shape[0]
m_test = x_test.shape[0]

plt.figure()
plt.plot(x, y, linestyle='', marker='x', color='r')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')


# =========== Part 2: Regularized Linear Regression Cost =============
theta = np.array([[1], [1]])
j, grad = linear_reg_cost_function(theta, np.hstack((np.ones((m, 1)), x)), y, 1)

print 'Cost at theta = [1 ; 1]:', j
print '(this value should be about 303.993192)'


# =========== Part 3: Regularized Linear Regression Gradient =============
theta = np.array([[1], [1]])
j, grad = linear_reg_cost_function(theta, np.hstack((np.ones((m, 1)), x)), y, 1)

print 'Gradient at theta = [1 ; 1]:', grad.ravel()
print '(this value should be about [-15.303016; 598.250744])'


# =========== Part 4: Train Linear Regression =============
# Train linear regression with lambda = 0
l = 0
theta = train_linear_reg(np.hstack((np.ones((m, 1)), x)), y, l)

pred = np.hstack((np.ones((m, 1)), x)).dot(theta)

plt.figure()
plt.plot(x, y, linestyle='', marker='x', color='r')
plt.plot(x, pred, linestyle='--', marker='', color='b')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

print np.hstack((np.ones((m, 1)), x)).dot(theta).shape
print (np.hstack((np.ones((m, 1)), x)).dot(theta) - y).shape


# =========== Part 5: Learning Curve for Linear Regression =============
l = 0
error_train, error_val = learning_curve(np.hstack((np.ones((m, 1)), x)), y, np.hstack((np.ones((m_val, 1)), x_val)), y_val, l)

plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')

print '# Training Examples / Train Error / Cross Validation Error'
for i in range(m):
    print '  {0:<19} {1:<13.8f} {2:<.8f}'.format(i + 1, error_train[i], error_val[i])


# =========== Part 6: Feature Mapping for Polynomial Regression =============
p = 8

x_poly = poly_features(x, p)
x_poly, mu, sigma = feature_normalize(x_poly)
x_poly = np.hstack((np.ones((m, 1)), x_poly))

x_poly_test = poly_features(x_test, p)
x_poly_test, dummy_mu, dummy_sigma = feature_normalize(x_poly_test, mu, sigma)
x_poly_test = np.hstack((np.ones((m_test, 1)), x_poly_test))

x_poly_val = poly_features(x_val, p)
x_poly_val, dummy_mu, dummy_sigma = feature_normalize(x_poly_val, mu, sigma)
x_poly_val = np.hstack((np.ones((m_val, 1)), x_poly_val))

print 'Normalized Training Example 1:'
print x_poly[0, :]


#  =========== Part 7: Learning Curve for Polynomial Regression =============
l = 0
theta = train_linear_reg(x_poly, y, l, iteration=500)

plt.figure()
plt.plot(x, y, linestyle='', marker='x', color='r')
plot_fit(np.min(x), np.max(x), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {})'.format(l))

error_train, error_val = learning_curve(x_poly, y, x_poly_val, y_val, l)
plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', marker='v', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(l))
plt.xlabel('Number of training examples')
plt.ylabel('Error')


# =========== Part 8: Validation for Selecting Lambda =============
lambda_vec, error_train, error_val = validation_curve(x_poly, y, x_poly_val, y_val)

plt.figure()
plt.plot(lambda_vec, error_train, color='b', label='Train')
plt.plot(lambda_vec, error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('lambda')
plt.ylabel('Error')

print '# lambda / Train Error / Validation Error'
for i in range(len(lambda_vec)):
    print '  {0:<8} {1:<13.8f} {2:<.8f}'.format(lambda_vec[i], error_train[i], error_val[i])


plt.show()
