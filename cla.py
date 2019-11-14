
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import pickle
import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array
import os
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
"""
image1 = cv2.imread('A1.jpg')
image = cv2.resize(image1, (28, 28))
#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0

image = img_to_array(image)
image = np.expand_dims(image, axis=0)
print(image.shape)
#image = image.flatten()
"""
input_alpha_index = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train = pd.read_csv('sign_mnist_train.csv')
train.drop('label', axis = 1, inplace = True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])
image = images.reshape(images.shape[0],28,28,1)[input_alpha_index]
image = np.expand_dims(image, axis=0)
image = image/255
model = load_model('SLI.h5')

lb = pickle.loads(open('lables.pickle', "rb").read())
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
alpha = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
print("Predicted output is: ")
print(alpha[label])
plt.imshow(images[input_alpha_index].reshape(28,28))
plt.show()
#print(model.summary())
