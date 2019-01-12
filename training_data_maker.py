# Programmer: Ben Morgenstern
# Credit to: pythonprogramming.net for tutorial on tensorflow/keras
# For: Hack Western 2018
# Purpose: This program formats the training data for use in training models


import os
import cv2
import pickle
import random
import numpy as np
from tqdm import tqdm

DATADIR = "C:\\Users\\Bens PC\\PycharmProjects\\hack_test\\git_data\\Apple"  # location of training data
CATEGORIES = ["Healthy", "Unhealthy"]  # categories of the data
IMG_SIZE = 70

def create_training_data():
    '''
    Creates list of training data in properly sized array form
    :return: a list of standard sized image arrays
    '''

    training_data = []

    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)  # create path to training data
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=healthy 1=unhealthy

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to standardize data size
                training_data.append([new_array, class_num])
            except OSError as e:
                print("OSError: Bad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("General exception", e, os.path.join(path,img))

    return training_data


training_data_formatted = create_training_data()
random.shuffle(training_data_formatted)

X = []  # feature set
y = []  # label set
for features, label in training_data_formatted:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # feature set must be reshapen as a numpy array

pickle_out = open("X.pickle","wb")  # saves X state for use later
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")  # saves y state for use later
pickle.dump(y, pickle_out)
pickle_out.close()
