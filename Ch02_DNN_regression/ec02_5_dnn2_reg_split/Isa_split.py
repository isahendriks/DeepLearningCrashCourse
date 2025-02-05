#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng, permutation, choice
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

def plot_mse_tv(mse_train, mse_val, smooth=11):
    """Plot MSE evolution validation data and training data during training."""
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    ax[0].plot(mse_train, c="tab:orange", label="train")
    ax[0].plot(mse_val, c="tab:green", label="validation")
    ax[0].set_xlabel("epoch", fontsize=24)
    ax[0].set_ylabel("MSE", fontsize=24)
    ax[0].tick_params(axis="both", which="major", labelsize=16)
    ax[0].legend(fontsize=16)
    ax[0].grid(True)
    
    ax[1].loglog(mse_train, c="tab:orange", label = "train")
    ax[1].loglog(mse_val, c="tab:green", label="validation")
    ax[1].set_xlabel("epoch", fontsize=24)
    ax[1].set_ylabel("MSE", fontsize=24)
    ax[1].tick_params(axis="both", which="major", labelsize=16)
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()
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

plot_pred_vs_gt(y_p, y_gt)
#plot_pred_2d(x, y_gt, y_p=dnn2_reg(wa, wb, x))

# %% Implement backpropagation batches
num_epochs = 10 ** 4
num_samples = len(x)

split = .70
num_samples_train = int(split*num_samples)
train_idx = choice(num_samples, num_samples_train, replace=False)

x_train = x[train_idx]
y_gt_train = y_gt[train_idx]
x_val = np.delete(x, train_idx, axis=0)  # Validation inputs.
y_gt_val = np.delete(y_gt, train_idx, axis=0)  # Validation ground truths.

eta = 0.1 # learning rate
num_batches = 7 #num_samples/25
batch_size = int(num_samples_train/num_batches)
mse_train = np.zeros((num_epochs,))
mse_val = np.zeros((num_epochs,))

#%%

for epoch in range(num_epochs):
    permuted_order_samples = permutation(num_samples_train)
    x_permuted = x_train[permuted_order_samples]
    y_gt_permuted = y_gt_train[permuted_order_samples]

    for batch_start in range(0, num_samples_train, batch_size):
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
    
    y_p_train = dnn2_reg(wa, wb , x_train)
    y_p_val = dnn2_reg(wa, wb, x_val)

    mse_train[epoch] = 1/2*np.mean((y_p_train - y_gt_train) ** 2)
    mse_val[epoch] = 1/2*np.mean((y_p_val - y_gt_val) ** 2)

    print(f"epoch = {epoch} MSE train = {mse_train[epoch]:.2f} MSE val = {mse_val[epoch]:.2f}")

#%% 
plot_mse(mse_train)
#%%
plot_mse(mse_val)