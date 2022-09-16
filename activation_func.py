from array import array
import numpy as np

def step_function(x:array):
    """
    setp function to output 1 when x is positive, or 0 else when
    Args:
        x (array): arbitrary real numpy array
    Returns:
        step: real array composed of 0 or 1
    """
    y = x > 0
    return y.astype(np.int)

def sigmoid(x:array):
    """
    sigmoid functino to output 
    Args:
        x (array): arbitrary real numpy array
    Returns:
        sigmoid: real array in range of 0 to 1
    """
    return 1/(1 + np.exp(-x))
    
def relu(x):
    return np.maximum(0, x)



