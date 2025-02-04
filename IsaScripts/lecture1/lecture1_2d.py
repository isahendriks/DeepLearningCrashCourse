#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%% Define functions

def load_data_2d(filename):
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
    plt.scatter(x[:,0], x[:,1], c=y_gt, s=50)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()

def plot_pred_2d(x, y_gt, y_p):
    plt.scatter(x[:,0], x[:,1], c=y_gt, s=50, label = "Ground truth")
    plt.scatter(x[:,0], x[:,1], c=y_p, s=100, marker = "x", label = "predicted")
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.legend()
    plt.show()

def neuron_clas_2d(w0,x):
    return (x@w0>0).astype(int)

def neuron_class_2d_bias(w0, b, x):
    return (w0@x +b> 0).astype(int)

#%% Load data
path = "C:\\Users\\is5046he\\Work Folders\\Documents\\GitHub\\NFFY314_DeepLearning\\DeepLearningCrashCourse-main\\Ch01_DNN_classification\\ec01_2_neuron_class_2d\\data_class_2d_clean.csv"
(x, y_gt) = load_data_2d(filename=path)

# %% Plot the data
plot_data_2d(x, y_gt)

#%% Import random number

rng = default_rng()
w = rng.standard_normal(size=(2,))
b = rng.standard_normal()

y_p = neuron_class_2d_bias(w,b,x)

plot_pred_2d(x, y_gt, y_p)

#%%
num_samples = len(x)
num_train_iterations = 100
eta = 0.1

for i in range(num_train_iterations):
    index = rng.integers(0,num_samples)
    x_selected = x[index]
    y_gt_selected = y_gt[index]

    y_p_selected = neuron_clas_2d(w, x_selected)

    error = y_p_selected - y_gt_selected

    w = w - eta * error * x_selected

    print(f"i={i} w0={float(w[0]):.2f} w1={float(w[1]):.2f} error={error[0]:.2f}")

#%%
plot_pred_2d(x, y_gt, y_p = neuron_clas_2d(w,x))