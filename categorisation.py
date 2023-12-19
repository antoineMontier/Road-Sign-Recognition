from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import sklearn
from keras.utils import to_categorical
import os
import sklearn.cluster
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, LearningRateScheduler
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def make_labels(directory, data=[], y_hat=[], label=0):
    for root, dirs, files in os.walk(directory):
        for file in files:
            img = matplotlib.image.imread(directory + file)
            
            # Ensure the image has 3 channels (RGB)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
                
            img = cv2.resize(img, (224, 224))
            data.append(img)
        y_hat = [label] * len(data)
    return np.array(data), np.array(y_hat)

def lr_schedule(epoch):
    lr = 1e-3  # Initial learning rate
    if epoch > 5:
        lr *= 1e-2  # Drop the learning rate by a factor of 100 after 5 epochs
    return lr

parent_folder = 'Training/augmentation2/'

a, y_a = [], []
a, y_a = make_labels(parent_folder + '/A/', data=a, y_hat=y_a, label=0)
print("a loaded")

b, y_b = [], []
b, y_b = make_labels(parent_folder + '/B/', data=b, y_hat=y_b, label=1)
print("b loaded")

c, y_c = [], []
c, y_c = make_labels(parent_folder + '/C/', data=c, y_hat=y_c, label=2)
print("c loaded")

d, y_d = [], []
d, y_d = make_labels(parent_folder + '/D/', data=d, y_hat=y_d, label=3)
print("d loaded")

e, y_e = [], []
e, y_e = make_labels(parent_folder + '/E/', data=e, y_hat=y_e, label=4)
print("e loaded")

f, y_f = [], []
f, y_f = make_labels(parent_folder + '/F/', data=f, y_hat=y_f, label=5)
print("f loaded")


print('a:', a.shape)
print('b:', b.shape)
print('c:', c.shape)
print('d:', d.shape)
print('e:', e.shape)
print('f:', f.shape)


X = np.vstack((a, b, c, d, e, f))
a = b = c = d = e = f = None
y = np.hstack((y_a, y_b, y_c, y_d, y_e, y_f)).reshape(-1, 1)
y_cat = to_categorical(y)

y = None

print(X.shape)
print(y_cat.shape)

classifier = Sequential()

# Convolutional Layer 1
classifier.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(BatchNormalization())

# Convolutional Layer 2
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(BatchNormalization())

# Convolutional Layer 3
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(BatchNormalization())

# Flatten Layer
classifier.add(Flatten())

# Dense Hidden Layer 1
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(BatchNormalization())

# Output Layer for Classification
classifier.add(Dense(6, activation='softmax'))

classifier.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

X = None
y_cat = None

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

classifier.compile( optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
lr_scheduler = LearningRateScheduler(lr_schedule)

# from keras.preprocessing.image import ImageDataGenerator
# 
# datagen = ImageDataGenerator(rescale=1./255)
# 
# train_generator = datagen.flow(X_train, y_train, batch_size=2)
# validation_generator = datagen.flow(X_test, y_test, batch_size=2)
# classifier.fit(train_generator, epochs=8, batch_size=16, validation_data=validation_generator, callbacks=[early_stopping])

classifier.fit(X_train, y_train, epochs=30, batch_size=3, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

classifier.save('categorizer-aug2-2-30ep.h5')