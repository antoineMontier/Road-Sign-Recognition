import os
import numpy as np
import cv2
import matplotlib
from matplotlib import image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import models
from tensorflow.keras import utils

# ===

# circle, kite, parallelogram, rectangle, rhombus, square, trapezoid, triangle
image_shape = (224, 224, 1)

# ===

def make_labels(directory, data=[], y_hat=[], label=0):
    for root, dirs, files in os.walk(directory):
        for file in files:
            img = np.reshape(cv2.cvtColor(matplotlib.image.imread(directory+file), cv2.COLOR_RGB2GRAY),image_shape)
            data.append(img)
        y_hat = [label] * len(data)
    return np.array(data), np.array(y_hat)

# ===

circle, y_circle = [], []
circle, y_circle = make_labels('shape-train/circle/', data=circle, y_hat=y_circle, label=0)
print("circle loaded")

no_circle, y_no_circle = [], []
no_circle, y_no_circle = make_labels('shape-train/not-circle/', data=no_circle, y_hat=y_no_circle, label=1)
print("no_circle loaded")

# ===

X = np.vstack((circle, no_circle))
y = np.hstack((y_circle, y_no_circle)).reshape(-1, 1)

# ===

classifier = Sequential()

c1 = Conv2D(48, (3,3), padding='same', input_shape = image_shape, activation = 'relu')
classifier.add(c1)
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(Dropout(0.5))
classifier.add(Flatten()) 
w1 = Dense(units = 32, activation = 'relu')
w2 = Dense(units = 16, activation = 'relu')
w3 = Dense(units = 4, activation = 'relu')
w_end = Dense(units = 2, activation = 'softmax')
classifier.add(w1)
classifier.add(Dropout(0.5)) 
classifier.add(w2)
classifier.add(Dropout(0.5)) 
classifier.add(w_end)
# classifier.summary()

# ===

classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# ===

y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

# ===

history = classifier.fit(X_train, y_train, batch_size=16, epochs=2, verbose=1, validation_data=(X_test, y_test))

from keras.models import load_model

classifier.save('shape-recognize-10epochs.h5')

scores = classifier.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]}")