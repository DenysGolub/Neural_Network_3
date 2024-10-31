from layer import Layer
class NeuralNetwork:
    def __init__(self, layer, activations):
        self.layer = list(layer)
        self.layers = [Layer(layer[i], layer[i + 1], activations[i]) for i in range(len(layer) - 1)]
        self.learning_rate = None

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
        
        for l in self.layers:
            l.learning_rate = value
        

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
                self.layers[i].backprop_hidden(self.layers[i + 1].gamma, self.layers[i + 1].weights)

        for layer in self.layers:
            layer.update_weights()


    def print_weights(self):
        for l in self.layers:
            print('-'*50)
            print(l.weights)
            print('-'*50)
            