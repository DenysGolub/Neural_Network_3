import numpy as np
import random
from activations import Activations

class Layer:
    def __init__(self, number_input, number_output, activation):
        self.number_input = number_input
        self.number_output = number_output

        self.outputs = np.zeros(number_output)
        self.inputs = np.zeros(number_input)
        self.weights = np.zeros((number_output, number_input))
        self.weights_delta = np.zeros((number_output, number_input))
        self.error_gradient = np.zeros(number_output)
        self.error = np.zeros(number_output)

        self.initialize_weights()
        self.activation = activation

    def initialize_weights(self):
        self.weights = np.random.rand(self.number_output, self.number_input) - 0.5

    def back_prop_output(self, expected):
        self.error = np.array(self.outputs) - np.array(expected)
        self.error_gradient = self.error * Activations.activate_back(self.outputs, self.activation)
        self.weights_delta = np.outer(self.error_gradient, self.inputs)

    def backprop_hidden(self, error_gradient_forward, weights_forward):
        self.error_gradient = np.dot(error_gradient_forward, weights_forward) * Activations.activate_back(self.outputs, self.activation)
        self.weights_delta = np.outer(self.error_gradient, self.inputs)

    def update_weights(self):
        self.weights -= self.weights_delta * self.learning_rate

    def feedforward(self, input_data):
        self.inputs = input_data
        self.outputs = np.dot(self.weights, self.inputs)
        self.outputs = Activations.activate_feed(self.outputs, self.activation)
        return self.outputs


    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
