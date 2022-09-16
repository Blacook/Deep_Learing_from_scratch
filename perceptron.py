import numpy as np


def AND(x1:float, x2:float):
    """
    AND gate to output 1 only when inputs are T/T

    Args:
        x1 (float): one float
        x2 (float): the other float

    Returns:
        binary: 0 is False, 1 is True
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7 
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1:float, x2:float):
    """
    NAND gate to output binary

    Args:
        x1 (float): _description_
        x2 (float): _description_

    Returns:
        _type_: _description_
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7 
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1:float, x2:float):
    """_summary_

    Args:
        x1 (float): _description_
        x2 (float): _description_

    Returns:
        _type_: _description_
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1:float, x2:float):
    """_summary_

    Args:
        x1 (float): _description_
        x2 (float): _description_

    Returns:
        _type_: _description_
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

