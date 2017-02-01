# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from display_data import display_data
from lr_cost_function import lr_cost_function
from one_vs_all import one_vs_all
from predict_one_vs_all import predict_one_vs_all


# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10

# =========== Part 1: Loading and Visualizing Data =============
print 'Loading and Visualizing Data...'

# Load Training Data
mat_data = sio.loadmat('ex3data1.mat')
x = mat_data['X']
y = mat_data['y']
m, n = x.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = x[rand_indices[0:100], :]
display_data(sel, padding=1)


# ============ Part 2a: Vectorize Logistic Regression ============
# Test case for lrCostFunction
theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
x_t = np.hstack((np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
lambda_t = 3

j, grad = lr_cost_function(theta_t, x_t, y_t, lambda_t)

print 'Cost:', j
print 'Expected cost: 2.534819'
print 'Gradients: \n', grad
print 'Expected gradients: \n', '0.146561\n-0.548558\n0.724722\n1.398003\n'

#  ============ Part 2b: One-vs-All Training ============
print 'Training One-vs-All Logistic Regression...'
l = 0.1
all_theta = one_vs_all(x, y, num_labels, l)


# ================ Part 3: Predict for One-Vs-All ================
pred = predict_one_vs_all(all_theta, x)
print 'Training Set Accuracy:', np.mean(pred == y) * 100


plt.show()
