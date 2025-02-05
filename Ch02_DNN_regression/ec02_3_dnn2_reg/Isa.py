#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng
from loader import *
from plotting import *

#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return np.maximum(0,x)

def d_ReLU(x):
    return np.where(x > 0, 1., 0.)

def dnn2_reg(wa, wb, x):
    return ReLU(x@wa)@wb

#%% 
# filename = "data_reg_2d_linear_clean.csv"
filename = "data_reg_2d_nonlinear.csv"

(x, y_gt) = load_data(filename)
plot_data_2d(x, y_gt)

#%% Starting vals
num_neurons = 3

rng = default_rng()
wa = rng.standard_normal(size=(2, num_neurons))  # Input weights layer 1.
wb = rng.standard_normal(size=(num_neurons, 1))  # Input weights layer 2.
y_p = dnn2_reg(wa, wb, x)

plot_pred_2d(x, y_gt, y_p=dnn2_reg(wa, wb, x))

#%% Implement backpropagation
num_train_iterations = 10000
num_samples = len(x)
eta = 0.01 # learning rate

for i in range(num_train_iterations):
    # Select random sample
    selected = rng.integers(0, num_samples)
    x_selected = np.reshape(x[selected], (1, -1))
    y_gt_selected = np.reshape(y_gt[selected], (1, -1))

    # Detailed neural network calculation
    x_selected_a = x_selected  # Input layer 1.
    p_a = x_selected_a @ wa  # Activation potential layer 1.
    y_selected_a = ReLU(p_a)  # Output layer 1.

    x_selected_b = y_selected_a  # Input layer 2.
    p_b = x_selected_b @ wb  # Activation potential layer 2.
    y_selected_b = p_b  # Output layer 2 (output neuron).
    
    y_p_selected = y_selected_b
    
    # Update weights
    error = y_p_selected - y_gt_selected

    delta_b = error * 1
    wb = wb - eta * delta_b * np.transpose(x_selected_b)

    delta_a = np.sum(wb * delta_b, axis=1) * d_ReLU(p_a)
    wa = wa - eta * delta_a * np.transpose(x_selected_a)

    if i % 100 == 0:
        print(f"{i} y_p={y_p_selected[0, 0]:.2f} error={error[0, 0]:.2f}")

#%% 
plot_pred_2d(x, y_gt, y_p = dnn2_reg(wa, wb, x))

#%% Batch implementation

#%% Implement backpropagation
num_train_iterations = 1000
num_samples = len(x)
eta = 0.01 # learning rate
batch_size = 10

for i in range(num_train_iterations):
    
    y_p_selected_list = np.full(batch_size, np.nan)

    for j in range(batch_size):
        # Select random sample
        selected = rng.integers(0, num_samples)
        x_selected = np.reshape(x[selected], (1, -1))
        y_gt_selected = np.reshape(y_gt[selected], (1, -1))

        # Detailed neural network calculation
        x_selected_a = x_selected  # Input layer 1.
        p_a = x_selected_a @ wa  # Activation potential layer 1.
        y_selected_a = ReLU(p_a)  # Output layer 1.

        x_selected_b = y_selected_a  # Input layer 2.
        p_b = x_selected_b @ wb  # Activation potential layer 2.
        y_selected_b = p_b  # Output layer 2 (output neuron).
        
        y_p_selected_list[j] = y_selected_b
    
    y_p_selected = float(np.mean(y_p_selected_list))

    # Update weights
    error = y_p_selected - y_gt_selected

    delta_b = error * 1
    wb = wb - eta * delta_b * np.transpose(x_selected_b)

    delta_a = np.sum(wb * delta_b, axis=1) * d_ReLU(p_a)
    wa = wa - eta * delta_a * np.transpose(x_selected_a)

    if i % 100 == 0:
        print(f"{i} y_p={y_p_selected[0, 0]:.2f} error={error[0, 0]:.2f}")

# %%
