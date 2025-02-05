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

def neuron_reg_1d(w0, x):
    """Artificial neuron for 1D regression."""
    return w0 * x

def plot_data_1d(x, y_gt):
    plt.scatter(x, y_gt, c="k", s=50)
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.grid()
    plt.show()

def plot_pred_1d(x, y_gt, y_p):
    """Plot 1D data and predictions."""
    plt.scatter(x, y_gt, s=20, c="k", label="ground truth")
    plt.scatter(x, y_p, s=100, c="tab:orange", marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()

#%%
path ="C:/Users/is5046he/Work Folders/Documents/GitHub/DeepLearningCrashCourse/Ch02_DNN_regression/ec02_1_neuron_reg_1d/data_reg_1d_clean.csv"
(x, y_gt) = load_data(path)

path ="C:/Users/is5046he/Work Folders/Documents/GitHub/DeepLearningCrashCourse/Ch02_DNN_regression/ec02_1_neuron_reg_1d/data_reg_1d_clean_test.csv"
(x_test, y_gt_test) = load_data(path)

#%%

num_samples = len(x)
num_train_iterations = 100
eta = 0.1

rng = default_rng()
w0 = rng.standard_normal()

y_p = neuron_reg_1d(w0, x)

for i in range(num_train_iterations):
    selected = rng.integers(0,num_samples)
    x0_selected = x[selected]
    y_gt_selected = y_gt[selected]

    y_p_selected = neuron_reg_1d(w0, x0_selected)

    error = y_p_selected - y_gt_selected

    w0 = w0 - eta * error * x0_selected
    print(f"i={i} w0={w0[0]:.2f} error = {error[0]:.2f}")

#%% 
plot_pred_1d(x, y_gt, y_p)

#%% More complex data

path ="C:/Users/is5046he/Work Folders/Documents/GitHub/DeepLearningCrashCourse/Ch02_DNN_regression/ec02_1_neuron_reg_1d/data_reg_1d_nonlinear.csv"
(x, y_gt) = load_data(path)
plot_data_1d(x, y_gt)

