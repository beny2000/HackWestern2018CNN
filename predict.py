# Programmer: Ben Morgenstern
# Credit to: pythonprogramming.net for tutorial on tensorflow/keras
# For: Hack Western 2018
# Purpose: This program is used to test the prediction of the model for a single image, simulating main use case

import cv2
import tensorflow as tf

CATEGORIES = ["Healthy", "Unhealthy"]  # converts prediction num to string value

def prepare(filepath):
    '''
    Prepares test image to run through model. Resizes and converts to array
    :param filepath:
    :return:
    '''

    IMG_SIZE = 70

    img_array = cv2.imread(filepath)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's standardized sizing

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with properly shaped for tf.

model = tf.keras.models.load_model("C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\models\\appleCNN-1543154559.model")
prediction = model.predict([prepare('apple.JPG')])

print(CATEGORIES[int(prediction[0][0])])  # output of the model is the index of the models prediction

