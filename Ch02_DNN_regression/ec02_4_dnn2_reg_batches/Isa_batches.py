
#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng, permutation
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
    return sigmoid(x@wa)@wb

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

#plot_pred_2d(x, y_gt, y_p=dnn2_reg(wa, wb, x))

# %% Implement backpropagation batches
num_epochs = 10 ** 4
num_samples = len(x)
eta = 0.1 # learning rate
num_batches = num_samples/25
batch_size = int(num_samples/num_batches)
mse_train = np.zeros((num_epochs,))

for epoch in range(num_epochs):
    permuted_order_samples = permutation(num_samples)
    x_permuted = x[permuted_order_samples]
    y_gt_permuted = y_gt[permuted_order_samples]

    for batch_start in range(0, num_samples, batch_size):
        dwa = np.zeros(wa.shape)
        dwb = np.zeros(wb.shape)

        for selected in range(batch_start, batch_start + batch_size):
            x_selected = reshape(x_permuted[selected], (1, -1))
            y_gt_selected = reshape(y_gt_permuted[selected], (1, -1))

            # Detailed neural network calculation
            x_selected_a = x_selected  # Input layer 1.
            p_a = x_selected_a @ wa  # Activation potential layer 1.
            y_selected_a = sigmoid(p_a)  # Output layer 1.

            x_selected_b = y_selected_a  # Input layer 2.
            p_b = x_selected_b @ wb  # Activation potential layer 2.
            y_selected_b = p_b  # Output layer 2 (output neuron).
            
            y_p_selected = y_selected_b
            error = y_p_selected - y_gt_selected

            delta_b = error * 1
            dwb = dwb - eta * delta_b * np.transpose(x_selected_b)

            delta_a = np.sum(wb * delta_b, axis=1) * d_sigmoid(p_a)
            dwa = dwa - eta * delta_a * np.transpose(x_selected_a)
        
        wa = wa + dwa/batch_size
        wb = wb + dwb/batch_size
    
    y_p = dnn2_reg(wa, wb , x)
    mse_train[epoch] = 1/2*np.mean((y_p - y_gt) ** 2)

    print(f"epoch = {epoch} MSE = {mse_train[epoch]:.2f}")

plot_pred_vs_gt(y_gt, y_p=dnn2_reg(wa, wb, x))

plot_mse(mse_train)

# %%
plot_pred_2d(x, y_gt, y_p=dnn2_reg(wa, wb, x))
#%%
plot_pred_vs_gt(y_gt, y_p=dnn2_reg(wa, wb, x))
#%%
plot_mse(mse_train)