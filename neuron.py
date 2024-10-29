import numpy as np
from activations import Activations as activ
class Neuron():
    def __init__(self, input_size, activation_function):
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0 
        
    def activation(self, x):
        if(self.activation_function == 'relu'):
            return activ.relu(x=x)
        elif (self.activation_function == 'sigmoid'):
            return activ.sigmoid(x=x)
    
    def activation_derivative(self,x):
        if(self.activation_function == 'relu'):
            return activ.relu_derivative(x=x)
        elif (self.activation_function == 'sigmoid'):
            return activ.sigmoid_derivative(x=x)
        
    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)
    
    def backward(self, gradients):
        pass