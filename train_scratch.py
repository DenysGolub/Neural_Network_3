# import cv2
# import os
# import numpy as np
# import tensorflow
# DATADIR = 'data'
# classes = os.listdir(DATADIR)

# X=[]
# y=[]
# for category in classes:
#     print(category)
#     path = os.path.join(DATADIR, category)
#     img_count = 100
#     while(img_count>0):
#         for img in os.listdir(path):
#                 try:
#                     img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_UNCHANGED)
#                     img_array = cv2.resize(img_array, (256, 256))
#                     gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

#                     #plt.imshow(gray_image)
#                     #print(gray_image.shape)
#                     X.append(np.array(gray_image))
#                     y.append(category)
#                     img_count -= 1
#                     #print(img_count)
#                 except Exception as e:
#                     pass
            
# y = [classes.index(label) for label in y]

# training_data = list(zip(X,y))
# print(training_data[0])

# import random
# random.shuffle(training_data)
# #print(training_data[500])

# X, y = zip(*training_data)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)



import numpy as np

from layer import Layer
import cv2
import os
DATADIR = 'data'
classes = os.listdir(DATADIR)

X=[]
y=[]

for category in classes:
    print(category)
    path = os.path.join(DATADIR, category)
    img_count = 1
    while(img_count>0):
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_UNCHANGED)
                img_array = cv2.resize(img_array, (256, 256))
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                #plt.imshow(gray_image)
                #print(gray_image.shape)
                X.append(np.array(gray_image))
                y.append(category)
                img_count-=1
                break
            except Exception as e:
                pass
            break
        break

from neural_network import NeuralNetwork

data = np.array([
    [1.0, 1.0],
    [9.4, 6.4],
    [2.5, 2.1],
    [8.0, 7.7],
    [0.5, 2.2],
    [7.9, 8.4],
    [7.0, 7.0],
    [2.8, 0.8],
    [1.2, 3.0],
    [7.8, 6.1]
])

y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])  # Target labels

layers = [2, 25, 10, 1]  # Hidden layers (input, [hidden], output)
activations = ["relu", "tanh", "tanh", "tanh", "sigmoid"]
nn = NeuralNetwork(layers, activations)
nn.learning_rate = 0.1
print(len(data[0]))


max_epoch = 10000
has_error = True
epoch = 0
import math
nn.print_weights()


while has_error and epoch < max_epoch:
    total_loss = 0
    has_error = False

    for i in range(len(data)):
        nn.feed_forward(data[i])

        network_output = nn.layers[-1].outputs[0]
        predicted = 1.0 / (1.0 + math.exp(-network_output))
        predicted_class = 1 if predicted >= 0.5 else 0

        total_loss += (predicted - y[i]) ** 2

        if predicted_class != y[i]:
            nn.backprop([y[i]])
            has_error = True

    # Calculate average loss if desired
    average_loss = total_loss / len(data)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}")
        print(f"Total Loss for Epoch {epoch}: {average_loss}")
    epoch += 1


nn.print_weights()

