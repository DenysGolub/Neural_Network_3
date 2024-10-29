
import numpy as np
from activations import Activations as act
class NeuralNetwork():
    learning_rate = 0.01
    layers = []
    outputs = []
    gradients = []
    def forward(self):
        for layer in self.layers:
            
            layer_outputs = []
            
            for neuron in layer.neurons:
                output = neuron.forward(self.inputs)
                layer_outputs.append(output)
            
            self.inputs = layer_outputs
            self.outputs.append(layer_outputs)  
            #print("Layer output:", self.inputs)
    
    def calculate_loss(self, true_labels):
        loss = 0
        for i, neur in enumerate(self.outputs[-1]):
            loss += (neur-true_labels[i])**2
        
        return loss
    
def backward(self, true_label):
    # Gradient for the last layer
    l_number = len(self.layers) - 1
    
    # Calculate error for the output layer
    error = self.calculate_loss_derivative(true_label, self.outputs[-1])  # Use derivative here

    while l_number > 0:
        layer = self.layers[l_number]
        previous_layer_outputs = self.outputs[l_number - 1]
        
        # Loop through neurons in the current layer
        for out_index, neuron in enumerate(layer.neurons):
            output = self.outputs[l_number][out_index]
            grad = error[out_index] * output * (1 - output)  # Gradient for the output layer neuron
            
            # Update weights for each neuron in the current layer
            for input_index, input_neuron in enumerate(self.layers[l_number - 1].neurons):
                input_val = previous_layer_outputs[input_index]
                input_neuron.weights -= self.learning_rate * grad * input_val
            
            # Compute error to propagate to the next layer
            # Add this gradient to be used in the next layer's backpropagation if l_number > 1
            if l_number > 1:
                for input_index, input_neuron in enumerate(self.layers[l_number - 1].neurons):
                    error[input_index] += neuron.weights[input_index] * grad
        
        l_number -= 1

    
                
                        
        
                                     

    def add_layer(self, layer):
        self.layers.append(layer)
        

    def predict(self, features):
        pass