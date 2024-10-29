from neuron import Neuron
class Layer():
    
    """
    Constructor input layer which defines a layer with number of neurons
    """
    def __init__(self, *args):
        
        if len(args) == 3 and isinstance(args[0], int):  # Hidden layers
            input_size, number_neurons, activation = args
            self.neurons = [Neuron(input_size, activation_function=activation) for _ in range(number_neurons)]
        else:
            raise ValueError("Invalid arguments provided to Layer constructor")