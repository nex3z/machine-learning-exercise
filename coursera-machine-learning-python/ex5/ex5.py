# Exercise 5: Regularized Linear Regression and Bias-Variance
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
print 'Part 1: Loading and Visualizing Data'
print 'Loading and Visualizing Data...'
mat_data = sio.loadmat('ex5data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
X_test = mat_data['Xtest']
y_test = mat_data['ytest'].ravel()
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()
m = X.shape[0]
m_val = X_val.shape[0]
m_test = X_test.shape[0]

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# =========== Part 2: Regularized Linear Regression Cost =============
print '\nPart 2: Regularized Linear Regression Cost'

theta = np.array([1, 1])
j, _ = linear_reg_cost_function(theta, np.hstack((np.ones((m, 1)), X)), y, 1)

print 'Cost at theta = [1 ; 1]:', j
print '(this value should be about 303.993192)'


# =========== Part 3: Regularized Linear Regression Gradient =============
print '\nPart 3: Regularized Linear Regression Gradient'

theta = np.array([1, 1])
_, grad = linear_reg_cost_function(theta, np.hstack((np.ones((m, 1)), X)), y, 1)

print 'Gradient at theta = [1 ; 1]:', grad.ravel()
print '(this value should be about [-15.303016; 598.250744])'


# =========== Part 4: Train Linear Regression =============
print '\nPart 4: Train Linear Regression'

# Train linear regression with lambda = 0
l = 0.0
theta = train_linear_reg(np.hstack((np.ones((m, 1)), X)), y, l)

pred = np.hstack((np.ones((m, 1)), X)).dot(theta)

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plt.plot(X, pred, linestyle='--', marker='', color='b')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# =========== Part 5: Learning Curve for Linear Regression =============
print '\nPart 5: Learning Curve for Linear Regression'

l = 0.0
error_train, error_val = learning_curve(np.hstack((np.ones((m, 1)), X)), y,
                                        np.hstack((np.ones((m_val, 1)), X_val)), y_val, l)

plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

print '# Training Examples / Train Error / Cross Validation Error'
for i in range(m):
    print '  {0:<19} {1:<13.8f} {2:<.8f}'.format(i + 1, error_train[i], error_val[i])


# =========== Part 6: Feature Mapping for Polynomial Regression =============
print '\nPart 6: Feature Mapping for Polynomial Regression'

p = 8

X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.hstack((np.ones((m, 1)), X_poly))

X_poly_test = poly_features(X_test, p)
X_poly_test, dummy_mu, dummy_sigma = feature_normalize(X_poly_test, mu, sigma)
X_poly_test = np.hstack((np.ones((m_test, 1)), X_poly_test))

X_poly_val = poly_features(X_val, p)
X_poly_val, dummy_mu, dummy_sigma = feature_normalize(X_poly_val, mu, sigma)
X_poly_val = np.hstack((np.ones((m_val, 1)), X_poly_val))

print 'Normalized Training Example 1:'
print X_poly[0, :]


#  =========== Part 7: Learning Curve for Polynomial Regression =============
print '\nPart 7: Learning Curve for Polynomial Regression'

l = 0.0
theta = train_linear_reg(X_poly, y, l, iteration=500)

plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plot_fit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {})'.format(l))
plt.show()

error_train, error_val = learning_curve(X_poly, y, X_poly_val, y_val, l)
plt.figure()
plt.plot(range(1, m + 1), error_train, color='b', marker='v', label='Train')
plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(l))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()


# =========== Part 8: Validation for Selecting Lambda =============
print '\nPart 8: Validation for Selecting Lambda'

lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, y_val)

plt.figure()
plt.plot(lambda_vec, error_train, color='b', label='Train')
plt.plot(lambda_vec, error_val, color='r', label='Cross Validation')
plt.legend(loc='upper right')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print '# lambda / Train Error / Validation Error'
for i in range(len(lambda_vec)):
    print '  {0:<8} {1:<13.8f} {2:<.8f}'.format(lambda_vec[i], error_train[i], error_val[i])
