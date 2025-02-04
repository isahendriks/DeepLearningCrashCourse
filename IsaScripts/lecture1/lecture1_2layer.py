#%% Import packages
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%% Define functions

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

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def d_sigmoid(x):
    """Derivative sigmoid function"""
    return(sigmoid(x) * (1 - sigmoid(x)))

def dnn2_class(wa, wb, x):
    x_a = x
    p_a = x_a @ wa
    y_a = sigmoid(p_a)

    x_b = y_a 
    p_b = x_b @ wb
    y_b = sigmoid(p_b)

    y_p = y_b

    return y_p

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

#%% Load data
path = "C:\\Users\\is5046he\\Work Folders\\Documents\\GitHub\\DeepLearningCrashCourse\\Ch01_DNN_classification\\ec01_3_dnn2_class\\"
filename = "data_class_2d_convex_clean_test.csv"

path_to_file = path + filename

(x, y_gt) = load_data(path_to_file)

#%% Plot data
plot_data_2d(x, y_gt)

#%% Define starting paramts
num_neurons = 6 # number of neurons in the hidden layer

rng  = default_rng()
wa = rng.standard_normal(size=(2, num_neurons))
wb = rng.standard_normal(size=(num_neurons, 1))

plot_pred_2d(x, y_gt, y_p=dnn2_class(wa, wb, x))

#%% Training the 2 layer network with error backpropagation through the network
num_train_iterations = 100000
num_samples = len(x)
eta = 0.1 # learning rate

for i in range(num_train_iterations):
    index = rng.integers(0, num_samples)
    x_selected = np.reshape(x[index], (1,-1))
    y_gt_selected = np.reshape(y_gt[index], (1,-1))

    x_selected_a = x_selected
    p_a = x_selected_a @ wa
    y_selected_a = sigmoid(p_a)

    x_selected_b = y_selected_a
    p_b = x_selected_b @ wb
    y_selected_b = sigmoid(p_b)

    y_p_selected = y_selected_b
    
    d_error = y_p_selected - y_gt_selected

    delta_b = d_error*d_sigmoid(p_b)
    wb = wb - eta * delta_b  * np.transpose(x_selected_b)

    delta_a = np.sum(wb*delta_b, axis=1) * d_sigmoid(p_a)
    wa = wa - eta * delta_a * np.transpose(x_selected_a)

    #w = w - eta * d_error * x_selected
    if i%100:
        print(f"i ={i} yp={y_p_selected[0,0]:.2f} error={d_error[0][0]:.2f}")

#%% Plot trained data
plot_pred_2d(x, y_gt, y_p=dnn2_class(wa, wb, x))