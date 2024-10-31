import cv2
import os
import numpy as np
import tensorflow
#import matplotlib.pyplot as plt
#print(len(tensorflow.config.list_physical_devics('GPU')))
with tensorflow.device('/GPU:0'):


    DATADIR = 'data'
    classes = os.listdir(DATADIR)

    X=[]
    y=[]
    for category in classes:
        print(category)
        path = os.path.join(DATADIR, category)
        img_count = 100
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
                        img_count -= 1
                        #print(img_count)
                    except Exception as e:
                        pass
                
    y = [classes.index(label) for label in y]

    training_data = list(zip(X,y))
    print(training_data[0])

    import random
    random.shuffle(training_data)
    #print(training_data[500])

    X, y = zip(*training_data)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


    from keras import callbacks, layers, models, optimizers, regularizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.losses import SparseCategoricalCrossentropy

    # Convert data to numpy arrays if needed
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)


    model = models.Sequential([
        layers.Input(shape=(256, 256, 1)),
        layers.Rescaling(1./255),
        
        # Add regularization to Conv2D layers and increase dropout
        layers.Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),
        
        layers.Conv2D(16, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(25, activation='softmax')
    ])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-10)

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=20,  # Increase epochs for better convergence
        callbacks=[reduce_lr, early_stopping]
    )

    model.save('arc_model.keras')
