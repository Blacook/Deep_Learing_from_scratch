import sys, os
sys.path.appned(os.pardir)
from common.functions import *
from common.gradients import numerical_grad

class TwoLayerNet:
    """
    Two Layer Neural Network
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int, weight_init_std:float=0.1):
        """
        initialize the prameters
        Args:
            input_size  (int): the number of input nuerons
            hidden_size (int): the number of hidden nuerons
            output_size (int): the number of output nuerons
            weight_init_std (float, optional): _description_. Defaults to 0.1.
        Selfs:
            params: the weight parameters
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)
        
        
    def predict(self, x:np.ndarray):
        """
        To predict or recognize the image data
        Args:
            x (np.ndarray): input image data
        Returns:
            y (np.ndarray): prediction score between 0 and 1
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    
    def loss(self, x:np.ndarray, t:np.ndarray):
        """
        To calculate the value of loss function 
        Args:
            x (np.ndarray): image data
            t (np.ndarray): correct answer labels
        Returns:
            loss(np.ndarray): the value of loss function; cross_entropy_error for multiclass identifier 
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)
        
        
    def accuracy(self, x:np.ndarray, t:np.ndarray):
        """
        To calculate the value of accuracy
        Args:
            x (np.ndarray): _description_
            t (np.ndarray): _description_

        Returns:
            accuracy(np.ndarray): the value of accuracy; the rate of accurate predictions 
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    
    def numerical_grad(self, x:np.ndarray, t:np.ndarray):
        """
        To calculate the gradients of the weight parameters; W, b
        Args:
            x (np.ndarray): _description_
            t (np.ndarray): _description_

        Returns:
            grads(dict): the gradients of the weight parameters; W, b 
        """
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_grad(loss_W, self.params['W1'])
        grads['b1'] = numerical_grad(loss_W, self.params['b1'])
        grads['W2'] = numerical_grad(loss_W, self.params['W2'])
        grads['b2'] = numerical_grad(loss_W, self.params['b2'])
        
        return grads
    
    #def grad(self, x:np.ndarray, t:np.ndarray):
        
    
