import numpy as np

from math import exp, log

# Compute log loss from observed y, computed y_p
def log_loss_function(y, y_p):
    return - ( y * np.log(y_p) + ( 1 - y) * np.log( 1 - y_p))

def sigmoid(x):
    return 1 / (1 + exp(-x))
