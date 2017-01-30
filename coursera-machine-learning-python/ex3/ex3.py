# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from display_data import display_data


# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10

# Part 1: Loading and Visualizing Data
print 'Loading and Visualizing Data...'
mat_data = sio.loadmat('ex3data1.mat')
x = mat_data['X']
y = mat_data['y']
m, n = x.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = x[rand_indices[0:100], :]
display_data(sel, padding=1)
# plt.show()
