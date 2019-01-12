# Programmer: Ben Morgenstern
# Credit to: pythonprogramming.net for tutorial on tensorflow/keras
# For: Hack Western 2018
# Purpose: This program is used to test the prediction of the model for a single image, simulating main use case

import os
import cv2
import tensorflow as tf
from random import shuffle


CATEGORIES = ["Healthy", "Unhealthy"]  # converts prediction num to string value

DATADIR = ""  # location of testing data

testing_data = []
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # creates path to imgs

    for img in os.listdir(path):
        if category == 'Healthy': # if image is in the healthhy folder then labels in as healthy
            file = 'C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\git_data\\Apple_testting_data\\Healthy\\' + img

        elif category == 'UnHealthy':
            file = 'C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\git_data\\Apple_testting_data\\UnHealthy\\' + img

        else:
            raise Exception("Bad File: ", img)  # incase issues arise

        testing_data.append([file, category])

shuffle(testing_data)

def prepare(filepath):
    '''
    Prepares test image to run through model. Resizes and converts to array
    :param filepath:
    :return:
    '''

    IMG_SIZE = 70  # 50 in txt-based

    img_array = cv2.imread(filepath)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's standard sizing

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image shaped for tf.


# Initialize testing variables
# best model --> C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\models\\appleCNN-1543157159.model
model = tf.keras.models.load_model("C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\models\\appleCNN-1543157159.model")
correct = []
wrong = []

for i in testing_data:
    prediction = model.predict([prepare(i[0])])

    if CATEGORIES[int(prediction[0][0])] == i[1]:  # if model predication matches img label then mark img as a correct prediction, else mark wrong
        correct.append(i)

    else:
        wrong.append(i)

    print(i[0], CATEGORIES[int(prediction[0][0])])

total_imgs = len(wrong) + len(correct)
print('Total Images: %f' % total_imgs)
print('Total Correct Predictions: %f\t Percentage correct: %f ' % (len(correct), len(correct)/total_imgs))
print('Total Wrong Predictions: %f\t Percentage wrong: %f' % (len(wrong), len(wrong)/total_imgs))
