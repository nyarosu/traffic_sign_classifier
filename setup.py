# Importing all dependancies. Info about each is in README.
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import random
import pickle
import os

# Divide our data into training (60%), validation (20%) and testing (20%)
with open("./dataset/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./dataset/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./dataset/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)

# Features are the inputs, labels are the outputs. We're assigning the inputs and outputs from our pickle files to variables that are easier to use.
x_train, y_train = train['features'], train['labels']
x_validation, y_validation = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

# Now we need to convert the images to grayscale and normalize them, after shuffling the data (this is to make sure the neural network doesn't learn the order of the images)
x_train, y_train = shuffle(x_train, y_train)

x_train_gray = np.sum(x_train/3, axis = 3, keepdims= True)
x_test_gray = np.sum(x_test/3, axis = 3, keepdims= True)
x_validation_gray = np.sum(x_validation/3, axis = 3, keepdims= True)

x_train_gray_norm = (x_train_gray - 128)/128
x_test_gray_norm = (x_test_gray - 128)/128
x_validation_gray_norm = (x_validation_gray - 128)/128

# Now that our data is ready, we can create the model using TensorFlow and Keras. We build our model piece by piece (layer by layer), but first we need to create a model that
# we can add the layers to. An annotated visual copy of the model is available in the README file of this repository.
CNN = models.Sequential()

CNN.add(layers.Conv2D(6, (5,5), activation= 'relu', input_shape = (32, 32, 1)))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Dropout(0.2)) # We drop out some neurons because neurons can become co-dependant on each other, hindering their performance.

CNN.add(layers.Conv2D(16, (5,5), activation= 'relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())

CNN.add(layers.Dense(120, activation='relu'))
CNN.add(layers.Dense(84, activation = 'relu'))
CNN.add(layers.Dense(43, activation = 'softmax')) # Output

# Now we compile and train our model. We compile our model with the Adam optimizer, as it's considered one of the best adaptive optimizers. We use sparse for our loss function as
# we have 43 different classes - we would use binary if we just had 2, and our only metric is accuracy as that's what we're assessing our model on.

CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = CNN.fit(x_train_gray_norm, y_train, batch_size = 500, epochs = 75, verbose = 1, validation_data = (x_validation_gray_norm, y_validation))
# After experimentation, 75 epochs seems to be the sweet spot. Any higher, and the model overfits - the validation loss increases, although accuracy continues to increase, which
# suggests overfitting. We have 98% accuracy and 0.44 validation loss at this "sweet spot". Now we'll use the test data - data that the model has never seen yet, to measure its true
# accuracy. But first, we can now save our model to disk.


blind_score = CNN.evaluate(x_test_gray_norm, y_test)
print("Test Accuracy: {}".format(blind_score[1]))

CNN.predict_classes()
CNN.save('./')