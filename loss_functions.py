import numpy as np
class LossFunctions():
    
    def __init__(self):
        pass
    
    @staticmethod
    def mae(expected, outputs):
        return (np.array(outputs) - np.array(expected))/len(expected)
    
    @staticmethod
    def cross_entropy(expected, outputs, epsilon=1e-12):
        outputs = np.clip(outputs, epsilon, 1. - epsilon)  
        return -np.sum(expected * np.log(outputs))
    
    @staticmethod
    def cross_entropy_loss_gradient(actual_labels, predicted_probs):
        num_samples = actual_labels.shape[0]
        gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples
        return gradient
    
    @staticmethod
    def calculate_loss(expected, outputs, loss):
        if(loss == 'mae'):
            return LossFunctions.mae(expected, outputs)
        elif (loss == 'cross_entropy'):
            return LossFunctions.cross_entropy(expected, outputs)
        
        
        