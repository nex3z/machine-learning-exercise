# Exercise 4: Neural Network Learning
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt

from display_data import display_data
from nn_cost_function import nn_cost_function
from sigmoid_gradient import sigmoid_gradient
from rand_initialize_weights import rand_initialize_weights
from predict import predict


# Setup the parameters you will use for this exercise
input_layer_size = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)


# =========== Part 1: Loading and Visualizing Data =============
print 'Loading and Visualizing Data...'

# Load Training Data
mat_data = sio.loadmat('ex4data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
m, n = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
plt.figure()
display_data(X[rand_indices[0:100], :], padding=1)
plt.show()


# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized neural network parameters.
print 'Loading Saved Neural Network Parameters...'

# Load the weights into variables theta_1 and theta_2
mat_param = sio.loadmat('ex4weights.mat')
theta_1 = mat_param['Theta1']
theta_2 = mat_param['Theta2']

# Unroll parameters
params_trained = np.hstack((theta_1.flatten(), theta_2.flatten()))

# ================ Part 3: Compute Cost (Feedforward) ================
print 'Feedforward Using Neural Network...'
l = 0.0
j, _ = nn_cost_function(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print 'Cost at parameters (loaded from ex4weights):', j, '(this value should be about 0.287629)'


# =============== Part 4: Implement Regularization ===============
print 'Checking Cost Function (w/ Regularization)...'
l = 1.0
j, _ = nn_cost_function(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print 'Cost at parameters (loaded from ex4weights):', j, '(this value should be about 0.383770)'


# ================ Part 5: Sigmoid Gradient  ================
print 'Evaluating sigmoid gradient...'
g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print 'Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:'
print g


# ================ Part 6: Initializing Parameters ================
print 'Initializing Neural Network Parameters...'
initial_theta_1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta_2 = rand_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = np.hstack((initial_theta_1.ravel(), initial_theta_2.ravel()))


# =============== Part 7: Implement Backpropagation ===============
print 'Checking Backpropagation...'


# =============== Part 8: Implement Regularization ===============
print 'Checking Backpropagation (w/ Regularization)...'
l = 3.0
debug_j, _ = nn_cost_function(params_trained, input_layer_size, hidden_layer_size, num_labels, X, y, l)
print 'Cost at (fixed) debugging parameters (w/ lambda = {}): {}'.format(l, debug_j)
print '(for lambda = 3, this value should be about 0.576051)'


# =================== Part 8: Training NN ===================
print 'Training Neural Network...'
l = 1.0
result = opt.minimize(fun=nn_cost_function, x0=initial_nn_params,
                      args=(input_layer_size, hidden_layer_size, num_labels, X, y, l),
                      method='TNC', jac=True, options={'maxiter': 150})
params_trained = result.x
Theta_1_trained = np.reshape(params_trained[0:(hidden_layer_size * (input_layer_size + 1)), ],
                             (hidden_layer_size, input_layer_size + 1))
Theta_2_trained = np.reshape(params_trained[(hidden_layer_size * (input_layer_size + 1)):, ],
                             (num_labels, hidden_layer_size + 1))


# ================= Part 9: Visualize Weights =================
print 'Visualizing Neural Network...'
plt.figure()
display_data(Theta_1_trained[:, 1:], padding=1)
plt.show()


# ================= Part 10: Implement Predict =================
pred = predict(Theta_1_trained, Theta_2_trained, X)
print 'Training Set Accuracy:', np.mean(pred == y) * 100
