from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import sys
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np


# Import trained model from disk.
CNN = keras.models.load_model('./')

img = image.load_img('./input.png', target_size=(32, 32))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
img_preprocessed_2 = np.sum(img_preprocessed/3, axis = 3, keepdims= True)
img_preprocessed_3 = (img_preprocessed_2 - 128)/128


prediction = CNN.predict(img_preprocessed_3)
classes_x=np.argmax(prediction,axis=1)
print(classes_x)