#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%% Define functions

def load_data_1d(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = np.asarray(data, dtype=float)
    
    x = data[:,0] # Input data
    y = data[:,1] # Ground truth
    return(x,y)

def plot_data_1d(x, y_gt):
    plt.scatter(x, y_gt, s=20, c="k")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()

def neuron_clas_1d(w0,x):
    return (w0*x > 0).astype(int)

def neuron_class_1d_bias(w0, b, x):
    return (w0*x +b> 0).astype(int)

#%% Load data
path = "C:\\Users\\is5046he\\Work Folders\\Documents\\GitHub\\NFFY314_DeepLearning\\DeepLearningCrashCourse-main\\Ch01_DNN_classification\\ec01_1_neuron_class_1d\\data_class_1d_clean.csv"

(x, y_gt) = load_data_1d(filename=path)

# %% Plot the data
print(x)
print(y_gt)

plot_data_1d( x, y_gt)

#%% Implement single neuron

rng = default_rng()
w0 = rng.standard_normal()
y_p = neuron_clas_1d(w0, x)

#%% Train the neuron 
num_samples = len(x)
num_train_iterations = 100
eta = 0.1

for i in range(num_train_iterations):

    index = rng.integers(0, num_samples)
    x0_selected = x[index]
    y_gt_selected = y_gt[index]

    y_p_selected = neuron_clas_1d(w0, x0_selected)

    error = y_p_selected - y_gt_selected # Caclulate error

    w0 = w0 -eta *error * x0_selected # Update neuron weight

#%% Train the nueron with a bias

#%% Train the neuron 
num_samples = len(x)
num_train_iterations = 100
eta = 0.1

w0 = rng.standard_normal()
b = rng.standard_normal()

for i in range(num_train_iterations):

    index = rng.integers(0, num_samples)
    x0_selected = x[index]
    y_gt_selected = y_gt[index]

    y_p_selected = neuron_class_1d_bias(w0, b, x0_selected)

    error = y_p_selected - y_gt_selected # Caclulate error

    w0 = w0 -eta *error * x0_selected # Update neuron weight
    b = b - eta*error
    print(f"i={i} w0 = {w0:.2f} b= {b:.2f} error = {error:.2f}")
