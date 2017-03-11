# Exercise 3-2: Neural Networks
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from display_data import display_data
from predict import predict


# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)


# =========== Part 1: Loading and Visualizing Data =============
print 'Loading and Visualizing Data...'
mat_data = sio.loadmat('ex3data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
m, n = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]
plt.figure()
display_data(sel, padding=1)
plt.show()


# ================ Part 2: Loading Parameters ================
print 'Loading Saved Neural Network Parameters...'
mat_param = sio.loadmat('ex3weights.mat')
Theta_1 = mat_param['Theta1']
Theta_2 = mat_param['Theta2']


# ================= Part 3: Implement Predict =================
pred = predict(Theta_1, Theta_2, X)
print 'Training Set Accuracy:', np.mean(pred == y) * 100

rp = np.random.permutation(m)
for i in range(m):
    print 'Displaying Example Image'
    display_data(X[rp[i],].reshape(1, n))

    pred = predict(Theta_1, Theta_2, X[rp[i],].reshape(1, n))
    print 'Neural Network Prediction: {} (digit {})'.format(pred, pred % 10)
    plt.show()
    s = raw_input('Paused - press enter to continue, q to exit: ')
    if s == 'q':
        break
