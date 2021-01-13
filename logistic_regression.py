
# ********************************************** #

#  Training Pipeline proceducte in pytorch

# 1 ) Design model (input, output size, fowrd pass)
# 2 ) construct loss optimizer
# 3 ) Training loop
#     - forward pass: copute prediction
#     - backward pass: gradients
#     - update weights

# ********************************************** #

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0 Prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_samples, n_features = X.shape

# print(n_features)


