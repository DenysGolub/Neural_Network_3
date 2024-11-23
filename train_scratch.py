import numpy as np
import data_helper
import cv2
from layer import FullyConnectedLayer, FlattenLayer, ActivationLayer, Softmax
from activations import Activation
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os

DATADIR = 'data/tumor'
classes = os.listdir(DATADIR)

dh=data_helper.DataHelper()
IMG_size = 16
IMG_size_display = 128



X, y = dh.load_data(img_counts=200, IMG_SIZE=IMG_size, DATADIR=DATADIR, binary_classification=False)

num_classes = len(classes)



X, y = np.array(X), np.array(y)
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

network = NeuralNetwork()
network.add_layer(FlattenLayer(input_shape=(X[0].shape)))
network.add_layer(FullyConnectedLayer(IMG_size*IMG_size, 512))
network.add_layer(ActivationLayer(Activation.relu, Activation.relu_derivative))
network.add_layer(FullyConnectedLayer(512, 256))
network.add_layer(ActivationLayer(Activation.relu, Activation.relu_derivative))
network.add_layer(FullyConnectedLayer(256, 128))
network.add_layer(ActivationLayer(Activation.relu, Activation.relu_derivative))
network.add_layer(FullyConnectedLayer(128, 64))
network.add_layer(ActivationLayer(Activation.relu, Activation.relu_derivative))
network.add_layer(FullyConnectedLayer(64, num_classes))
network.add_layer(Softmax(num_classes))

epochs = 10
learning_rate = 0.01


network.train(epochs=epochs, X_train=X_train, y_train=y_train)

from datetime import datetime

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

import pickle


network.set_classes(classes)

with open(f"models/network_{dt_string}_{IMG_size}px", 'wb') as file:
    pickle.dump(network, file)



y_pred = [np.argmax(network.predict(x)) for x in X_test]
y_true = [np.argmax(y) for y in y_test]

conf_matrix = confusion_matrix(y_true, y_pred)
print("Матриця плутанини:")
print(conf_matrix)
print("Звіт про класифікацію:")
print(classification_report(y_true, y_pred, target_names=classes))
network.confusion_matrix = conf_matrix

network.get_plot_confusion_matrix().show()
imp = dh.import_network(f"models/network_{dt_string}_{IMG_size}px")

accuracy = 0.0
true_pred = 0

classes = os.listdir(DATADIR)
for test, true in zip(X_test, y_test):
    image = cv2.resize(test, (IMG_size_display, IMG_size_display))
    pred = network.predict(test)[0]
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    
    if(idx==idx_true):
        true_pred += 1
    # plt.title(f'pred: {classes[int(idx)]}, prob: {pred[idx]}, true: {classes[int(idx_true)]}')
    # plt.imshow(image, cmap='binary')
    # plt.show()
    
print(f"Загальна кількість зображень: {len(X_test)}")
print(f"Кількість правильно класифікованих зображень: {true_pred}")
print(f"Точність: {true_pred/len(X_test)}")

