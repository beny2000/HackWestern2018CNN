# Programmer: Ben Morgenstern
# Credit to: pythonprogramming.net for tutorial on tensorflow/keras
# For: Hack Western 2018
# Purpose: This program tests for the optimal combinations of hidden dense layers, convolution layers and the layer size


import time
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

pickle_in = open("X.pickle","rb")  # imports X saved state
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")# imports y saved state
y = pickle.load(pickle_in)

X = X/255.0
count = 0
dense_layers = [0, 1]  # number of dense layers to test
layer_sizes = [32, 64, 128, 256]  # layer sizes to test
conv_layers = [1, 2, 3, 4]  # number of convolution layers to test
batch_size = 32
epochs = 1
validation = 0.3
loss_func = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            # Creates initial input layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Creates model with test number of conv layers
            for i in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            # Creates model with test number of dense layers
            for i in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # Final output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))  # Creates tensorboard to view model data

            model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)  # complies model

            model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=validation,
                      callbacks=[tensorboard])
