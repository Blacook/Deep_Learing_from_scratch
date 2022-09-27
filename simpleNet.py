import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradients import numerical_grad

class simpleNet:
    """
    
    """
    def __init__(self):
        """
        To initialize a weight matrix on the Gauss distribution
        """
        self.W = np.random.randn(2,3)
        
        
    def predict(self, x:np.ndarray):
        """

        Args:
            x (np.ndarray): input data

        Returns:
            dot: _description_
        """
        return np.dot(x, self.W)
    
    
    def loss(self, x:np.ndarray, t:np.ndarray):
        """

        Args:
            x (np.ndarray): actual or prediction data
            t (np.ndarray): true data to evaluate the prediction
        Returns:
            loss: _description_
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y ,t)    
        return loss
    
    
    def gds(self, x:np.ndarray, t:np.ndarray):
        """
        
        Args:
            x (np.ndarray): _description_
            t (np.ndarray): _description_
        """
        f = lambda w: self.loss(x, t)
        dW = numerical_grad(f, self.W)
        return dW

