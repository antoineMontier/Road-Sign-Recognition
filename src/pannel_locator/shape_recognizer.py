import os
import numpy as np
import cv2
import matplotlib
import random
from matplotlib import image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras import models
from tensorflow.keras import utils

# ===
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# ===

# circle, kite, parallelogram, rectangle, rhombus, square, trapezoid, triangle

# ===

def make_labels(directory, data=[], y_hat=[], label=0):
    for root, dirs, files in os.walk(directory):
        for file in files:
            img = np.reshape(cv2.cvtColor(matplotlib.image.imread(directory+file), cv2.COLOR_RGB2GRAY),(224, 224, 1))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            img = cv2.resize(img, (224, 224))

            data.append(img)
        y_hat = [label] * len(data)
    return np.array(data), np.array(y_hat)

# ===

circle, y_circle = [], []
circle, y_circle = make_labels('./../../shape-train/circle/', data=circle, y_hat=y_circle, label=0)
print("circle loaded")

no_circle, y_no_circle = [], []
no_circle, y_no_circle = make_labels('./../../shape-train/not-circle/', data=no_circle, y_hat=y_no_circle, label=1)
print("no_circle loaded")

# ===

X = np.vstack((circle, no_circle))
y = np.hstack((y_circle, y_no_circle)).reshape(-1, 1)

# ===

y_cat = to_categorical(y)
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

print("data normalized")

# ===

circle = None
y_circle = None
no_circle = None
y_no_circle = None
print("emptied not normalized data")


# ===

# Load a pre-trained MobileNetV2 model as a feature extractor
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

print("pre-trained model loaded")

# Create a custom head for classification
classifier = Sequential()
classifier.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
classifier.add(Dropout(0.5))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid'))  # Binary classification (circle or not circle)

print("custom model loaded")

# Combine the base model and the custom head
model = Sequential()
model.add(base_model)
model.add(classifier)

print("model merged")

# Set the layers in the base model as non-trainable (feature extractor)
for layer in base_model.layers:
    layer.trainable = False

print("made untrainable base model layers")

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("compiled model, ready to fit")

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=30, verbose=1, validation_data=(X_test, y_test))

model.save('shape-recognizerv3-30eh.h5')

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]}")