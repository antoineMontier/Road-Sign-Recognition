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
labels = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']
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
circle, y_circle = make_labels('shape-train/circle/', data=circle, y_hat=y_circle, label=labels.index('circle'))
print("circle loaded")

kite, y_kite = [], []
kite, y_kite = make_labels('shape-train/kite/', data=kite, y_hat=y_kite, label=labels.index('kite'))
print("kite loaded")

parallelogram, y_parallelogram = [], []
parallelogram, y_parallelogram = make_labels('shape-train/parallelogram/', data=parallelogram, y_hat=y_parallelogram, label=labels.index('parallelogram'))
print("parallelogram loaded")

rectangle, y_rectangle = [], []
rectangle, y_rectangle = make_labels('shape-train/rectangle/', data=rectangle, y_hat=y_rectangle, label=labels.index('rectangle'))
print("rectangle loaded")

rhombus, y_rhombus = [], []
rhombus, y_rhombus = make_labels('shape-train/rhombus/', data=rhombus, y_hat=y_rhombus, label=labels.index('rhombus'))
print("rhombus loaded")

square, y_square = [], []
square, y_square = make_labels('shape-train/square/', data=square, y_hat=y_square, label=labels.index('square'))
print("square loaded")

trapezoid, y_trapezoid = [], []
trapezoid, y_trapezoid = make_labels('shape-train/trapezoid/', data=trapezoid, y_hat=y_trapezoid, label=labels.index('trapezoid'))
print("trapezoid loaded")

triangle, y_triangle = [], []
triangle, y_triangle = make_labels('shape-train/triangle/', data=triangle, y_hat=y_triangle, label=labels.index('triangle'))
print("triangle loaded")

# ===

X = np.vstack((circle, kite, parallelogram, rectangle, rhombus, square, trapezoid, triangle))
y = np.hstack((y_circle, y_kite, y_parallelogram, y_rectangle, y_rhombus, y_square, y_trapezoid, y_triangle)).reshape(-1, 1)

# ===

classifier = Sequential()
# Convolution layer, with 32 random 3x3 kernels
c1 = Conv2D(48, (3,3), padding='same', input_shape = image_shape, activation = 'relu')
classifier.add(c1)
classifier.add(MaxPooling2D(pool_size=(3, 3)))
classifier.add(Dropout(0.5))
classifier.add(Flatten()) 
w1 = Dense(units = 32, activation = 'relu')
w2 = Dense(units = 16, activation = 'relu')
w3 = Dense(units = 4, activation = 'relu')
w_end = Dense(units = 8, activation = 'softmax')
classifier.add(w1)
classifier.add(Dropout(0.5)) 
classifier.add(w2)
classifier.add(Dropout(0.5)) 
classifier.add(w_end)
# classifier.summary()

# ===

classifier.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ===

y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

# ===

history = classifier.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_test, y_test))

from keras.models import load_model

classifier.save('shape-recognize-10epochs.h5')