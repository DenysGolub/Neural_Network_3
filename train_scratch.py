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

data=[
        [1.5, 0.3, 0.1, 4, 1],
        [2, 7, 4, 1, 2],
        [3.1, 0.9, 0.4, 34, 3],
        [3, 1, 6, 1, 4]
      
      ]

data = [[1.5, 0.3, 0.1, 4, 1]]

y=[1] 
#[X[0].reshape(1, 256*256)]
print(input)
layers = [5,3,3,1] #hidden layers (input, [hiddens], output)

nn = NeuralNetwork()
nn.learning_rate = 0.1
print(len(data[0]))

for i in range(0, len(layers)):
    if(i+1 >= len(layers)):
        break
    #print(f"Layer: input {layers[i]}| output {layers[i+1]}")
    
    if(i==len(layers)-1):
        nn.add_layer(Layer(layers[i], layers[i+1], 'sigmoid'))
    else: 
        nn.add_layer(Layer(layers[i], layers[i+1], 'relu'))  

# print(nn.layers[0].neurons)
# print(nn.layers[1].neurons)
# print(nn.layers[2].neurons)

# print('Layers added')
hasError = True
while(hasError):
    for i, feature in enumerate(data):
        nn.inputs = feature
        nn.forward()

        predicted = 1 / (1 + np.exp(-np.float32(nn.outputs[-1])))
        #print("Predicted:", predicted)

        predicted_class = 1 if predicted >= 0.5 else 0

        if predicted_class == y:
            hasError = False
        else:
            nn.backward(y)  
            hasError = True



