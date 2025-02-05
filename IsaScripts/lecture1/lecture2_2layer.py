#%%#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%%
def load_data(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = np.asarray(data, dtype=float)
    
    x = data[:,0:-1] # Input data
    num_samples = data.shape[0]
    y = np.reshape(data[:,-1], (num_samples,1)) # Ground truth
    return(x,y)

def plot_data_2d(x, y_gt):
    """Plot 2D data."""
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_pred_2d(x, y_gt, y_p):
    """Plot 2D data and predictions."""
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50, label="ground truth")
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100, marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()

def neuron_reg_2d(w, x):
    return x@w

#%% 
path ="C:/Users/is5046he/Work Folders/Documents/GitHub/DeepLearningCrashCourse/Ch02_DNN_regression/ec02_2_neuron_reg_2d/data_reg_2d_clean.csv"
(x, y_gt) = load_data(path)

#%%

plot_data_2d(x, y_gt)

#%% Ranodmly initialize weights

rng = default_rng()
w = rng.standard_normal(size=(2,))
y_p = neuron_reg_2d(w,x)

#%% Plot random prediction

plot_pred_2d(x, y_gt, y_p)

#%%
num_samples = len(x)
num_train_iterations = 100
eta = 0.1

for i in range(num_train_iterations):
    selected = rng.integers(0,num_samples)
    x0_selected = x[selected]
    y_gt_selected = y_gt[selected]

    y_p_selected = neuron_reg_2d(w0, x0_selected)

    error = y_p_selected - y_gt_selected

    w0 = w0 - eta * error * x0_selected
    print(f"i={i} w0={w0[0]:.2f} error = {error[0]:.2f}")
