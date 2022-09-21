import numpy as np

def numerical_diff(f:function, x:np.ndarray):
    """
    numerical differential for one variable
    Args:
        f (function): _description_
        x (array)    : _description_
    Returns:
        _type_: _description_
    """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
   
   
def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 -fxh2) / (2*h)
        x[idx] = tmp_val 
    
    return grad
         
    
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


