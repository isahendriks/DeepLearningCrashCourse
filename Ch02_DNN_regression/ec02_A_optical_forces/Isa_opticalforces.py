#%% Import packages
from os import path, system

if not path.exists("optical_forces_dataset"):
    system("git clone https://github.com/DeepTrackAI/optical_forces_dataset")

import numpy as np
#%% Define functions

#%% Import data