import random
import numpy as np


class Layer():
    """Base class for all layers"""
    
    def __init__():
        pass
    
       
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
    
class FullyConnectedLayer(Layer):
    
    """Represents fully-connected layer in neural network"""
        
    
    def __init__(self, input_size, output_size):
        """
        Constructor for fcl. Takes input_size as number of neurons in previous layer
        and output_size as number of output neurons in current layer.
        Weights and biases are randomly generating by Xavier generation
        """  
        self.input_size = input_size #set number of input neurons
        self.output_size = output_size #set number of output neurons 
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size) #randomly initialize weights normalized by Xavier
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size) #same thing as for weights but for bias
    
    
    def forward(self, input):
        
        """Feed forward in nn

        Returns:
            _type_: _description_
        """
        self.input = input #set input for layer
        return np.dot(input, self.weights) + self.bias #return weighted sum
    
    def backward(self, error):
        
        """Backprogapation in neural network. 

        Returns:
            _type_: _description_
        """
        input_error = np.dot(error, self.weights.T) #error that will go to the previous layer
        weights_error = np.dot(self.input.T, error) #delta W
        
        self.weights -= self._learning_rate * weights_error #update weights
        self.bias = self.learning_rate * error #update bias
        
        return input_error
    
class FlattenLayer(Layer):
    
    """Represents flatten layer of neural network. Flatten makes input shape to one column"""
    def __init__(self, input_shape):
        self.input_shape=input_shape
        
    def forward(self, input):
        return np.reshape(input, (1,-1))
    
    def backward(self, output_error):
        return np.reshape(output_error, self.input_shape)
        
        
class Softmax(Layer):
    def __init__(self, input_size):
        self.input_size=input_size
        
    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_error):
        out = np.tile(self.output.T, self.input_size) #creating smth familiar to jacobian
        jacobian_approx = np.identity(self.input_size) - out #here we already looked for situation w*(1-y) and w*y
        return self.output * np.dot(output_error, jacobian_approx) #return gradient


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_error):
        return output_error * self.activation_derivative(self.input)
    
class Conv2D(Layer):
    def __init__(self, input_shape, filter_size, num_filters):
        pass