# Programmer: Ben Morgenstern
# Credit to: pythonprogramming.net for tutorial on tensorflow/keras
# For: Hack Western 2018
# Purpose: This program is th emain program that trains the model using the layer data from find_optimal_CNN


import time
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten


# Initializes model variables
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)  # feature set

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)  # label set

X = X/255.0
conv_layer = 2
layer_size = 64
dense_layer = 0
batch_size = 32
epochs = 1
validation = 0.3
loss_func = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']


NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
print(NAME)

model = Sequential()

# Creates initial input layer
model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Creates the conv layers
for i in range(conv_layer-1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Creates Dense layers. Could remove since current data suggest no dense layers be used
for i in range(dense_layer):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Final output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation, callbacks=[tensorboard])

# Saves model by time stamp
time = int(time.time())
model.save('C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\models\\appleCNN-%d.model' % time)
print("Model Saved at C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\models\\appleCNN-%d.model" % time)
