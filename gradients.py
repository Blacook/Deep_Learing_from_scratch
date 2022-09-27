import numpy as np

def numerical_diff(f:object, x:np.ndarray):
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
   
   
def numerical_grad(f:object, x:np.ndarray):
    """

    Args:
        f (object): _description_
        x (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        # FIXME: IndexError: index 2 is out of bounds for axis 0 with size 2
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 -fxh2) / (2*h)
        #reset the temporary value of function
        x[idx] = tmp_val 
    
    return grad
         
         
def grad_descent(f:object, init_x:np.ndarray, lr:float=0.01, step_num:int=100):
    """
    gradient descent method to find the minimum of the function
    Args:
        f        (function): _description_
        init_x (np.ndarray): _description_
        lr          (float): _description_. Defaults to 0.01.
        step_num      (int): _description_. Defaults to 100.
    Returns:
        _type_: _description_
    """
    x = init_x
    for i in range(step_num):
        grad = numerical_grad(f, x)
        x -= lr * grad
        
    return x
        
        
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


