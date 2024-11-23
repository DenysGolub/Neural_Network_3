import numpy as np
import math
from layer import Layer
class NeuralNetwork:
    def __init__(self, layer, activations):
           # Store the layer configuration
        self.layer = layer[:]  # Copy layer list

        # Initialize the layers list
        self.layers = [
            Layer(layer[i], layer[i + 1], activations[i])
            for i in range(0, len(layer)-1)
        ]
        
        for i in range(0, len(layer)):
            print(i)
        self.learning_rate = None
        self._loss_function = None


    def fit(self, X,y, max_epoch):
        has_error = True
        epoch = 0
        while has_error and epoch < max_epoch:
            has_error = False
            total_loss = 0
            for i in range(len(X)):
                self.feed_forward(X[i])

                predicted = self.layers[-1].outputs
                # predicted = 1.0 / (1.0 + math.exp(-network_output))
                # predicted_class = 1 if predicted >= 0.5 else 0

                total_loss += sum((predicted - y[i]) ** 2)

                # print(predicted)
                # print(y[i])
                if str(predicted != y[i]):
                    self.backprop([y[i]])
                    has_error = True

            # Calculate average loss if desired
            average_loss = total_loss / len(X)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}")
                print(f"Total Loss for Epoch {epoch}: {average_loss}")
            epoch += 1

        
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
        
        for l in self.layers:
            l.learning_rate = value
        
    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value
        for l in self.layers:
            l.loss_function = value
        
    def predict(self, input):
        self.feed_forward(input)
        return self.layers[-1].outputs

    def feed_forward(self, input):
        self.layers[0].feedforward(input)
        for i in range(1, len(self.layers)):
            self.layers[i].feedforward(self.layers[i - 1].outputs)

        return self.layers[-1].outputs

    def backprop(self, expected):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].back_prop_output(expected)
            else:
                self.layers[i].backprop_hidden(self.layers[i + 1].error_gradient, self.layers[i + 1].weights)

        for layer in self.layers:
            layer.update_weights()


    def print_weights(self):
        for l in self.layers:
            print('-'*50)
            print(l.weights)
            print('-'*50)
            