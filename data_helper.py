import numpy as np
import random
import cv2
from layer import FullyConnectedLayer, FlattenLayer, ActivationLayer, Softmax
from activations import Activation
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
import pickle
class DataHelper():
    
    def __init__(self):
        pass

    def load_data(self, DATADIR='data', img_counts=100, IMG_SIZE=24, binary_classification=False):
        classes = os.listdir(DATADIR)

        X = []
        y= []
        for category in classes:
            print(category)
            img_count=img_counts
            path = os.path.join(DATADIR, category)
            while(img_count>0):
                for img in os.listdir(path):
                        try:
                            img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_UNCHANGED)
                            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                            gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                            X.append(np.array(gray_image))
                            import matplotlib.pyplot as plt
                            plt.imshow(img_array)
                            plt.show()
                            if(binary_classification==True):
                                y.append(category)
                            else:
                                y.append(self.one_hot_encode(classes.index(category), len(classes)))
                                
                            img_count -= 1
                        except Exception as e:
                            #print(e)
                            pass
                        
        data = list(zip(X, y))
        random.shuffle(data)
        X, y = zip(*data)
        return X,y
                    
    
    def one_hot_encode(self, class_index, len):
        return [1 if x == class_index else 0 for x in range(0, len)]

    def export_network(network, file_name):
        with open(f'models/{file_name}.pkl', 'wb') as file:
            pickle.dump(network, file)
            

    def import_network(self, file_name):
        loaded_network = None
        with open(file_name, 'rb') as file:
            loaded_network = pickle.load(file)
        return loaded_network

        
    @staticmethod
    def load_resized_gray_image(IMG_SIZE, path):
        img_array = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image/255.0
        
        return gray_image
