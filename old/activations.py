import math
import numpy as np

class Activations:
    @staticmethod
    def activate_feed(x, act):
        if act == "relu":
            return Activations.relu(x)
        elif act == "sigmoid":
            return Activations.sigmoid(x)
        elif act == "tanh":
            return [math.tanh(x_inp) for x_inp in x]
        elif act == "leakyrelu":
            return Activations.leaky_relu(x)
        elif act == "softmax":
            return Activations.softmax(x)
        return 0

    @staticmethod
    def activate_back(x, act):
        if act == "relu":
            return Activations.relu_derivative(x)
        elif act == "sigmoid":
            return Activations.sigmoid_derivative(x)
        elif act == "tanh":
            return Activations.tanh_derivative(x)
        elif act == "leakyrelu":
            return Activations.leaky_relu_derivative(x)
        elif act == "softmax":
            return Activations.softmax_derivative(x)
        return 0

    @staticmethod
    def tanh_derivative(x):
        return [1 - (x_inp * x_inp) for x_inp in x]

    @staticmethod
    def relu(x):
        out = []
        for x_inp in x:
            out.append(max(0, float(x_inp)))
        return out
    @staticmethod
    def relu_derivative(x):
        return [1 if x_inp > 0 else 0 for x_inp in x]

    @staticmethod
    def sigmoid(x):
        return [1.0 / (1.0 + math.exp(-x_inp)) for x_inp in x]

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return [x_inp if x_inp > 0 else alpha * x_inp for x_inp in x]

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return [1 if x_inp > 0 else alpha for x_inp in x]

    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_value = Activations.sigmoid(x)
        return [value * (1 - value) for value in sigmoid_value]

    @staticmethod
    def softmax(probabilities):
        exp_probs = np.exp(probabilities - np.max(probabilities))
        return exp_probs / exp_probs.sum()

    @staticmethod
    def softmax_derivative(probabilities):
        s = probabilities.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
