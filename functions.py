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
    sigmoid function to output 
    Args:
        x (array): arbitrary real numpy array
    Returns:
        sigmoid: real array in range of 0 to 1
    """
    return 1/(1 + np.exp(-x))

    
def relu(x:array):
    """
    ReLu function to output
    Args:
        x (array): _description_
    Returns:
        _type_: _description_
    """
    return np.maximum(0, x)


def identity_func(x:array):
    """
    identical function to output the input
    Args:
        x (array): _description_
    Returns:
        _type_: _description_
    """
    return x

def softmax(x:array):
    """
    softmax function to output 
    Args:
        x (array): _description_

    Returns:
        _type_: _description_
    """
    e = np.max(x)
    exp_x = np.exp(x - e) # prevention from overflow 
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    
    return y


def sum_squared_error(y:np.ndarray, t:np.ndarray):
    """
    sum of squared error
    Args:
        y (array): _description_
        t (array): _description_

    Returns:
        _type_: _description_
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y:np.ndarray, t:np.ndarray):
    """
    cross entropy . This is available if input is a batch.
    Args:
        y (array): _description_
        t (array): _description_
    Returns:
        _type_: _description_
    """
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


