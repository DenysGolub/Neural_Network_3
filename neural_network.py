import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from loss_functions import LossFunctions
class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.classes = []
        self.confusion_matrix = None
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def set_classes(self, classes):
        self.classes = classes
            
    def train(self, X_train, y_train, epochs=20, learning_rate=0.01):
        for epoch in range(epochs):
            error = 0
            for x, y_true in zip(X_train, y_train):
                output = x
                for layer in self.layers:
                    layer.learning_rate = learning_rate
                    output = layer.forward(output)
                
                error += LossFunctions.cross_entropy(y_true, output)

                output_error = LossFunctions.cross_entropy_loss_gradient(y_true, output)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error)
            
            error /= len(X_train)
            print('%d/%d, error=%f' % (epoch + 1, epochs, error))

    def predict(self, input):
            output = input
            for layer in self.layers:
                output = layer.forward(output)
                
            return output            

    
    def get_plot_confusion_matrix(self):
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Матриця плутанини')
        plt.xlabel('Передбачені класи')
        plt.ylabel('Реальні класи')
        
        return fig