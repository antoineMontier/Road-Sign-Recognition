import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define your class labels based on the folder names
class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
num_classes = len(class_labels)

# Image dimensions
img_height, img_width = 224, 224

# Path to your dataset
dataset_path = './../../Training/augmentation3/'

# Function to load images and labels
def load_dataset(dataset_path, class_labels):
    images = []
    labels = []
    for label, class_name in enumerate(class_labels):
        class_folder = os.path.join(dataset_path, class_name)
        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (img_width, img_height))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Loading the dataset
images, labels = load_dataset(dataset_path, class_labels)

print("raw data loaded")

# Normalize pixel values to be between 0 and 1
images = images / 255.0

print("normalized")

# Convert class vectors to binary class matrices
labels = to_categorical(labels, num_classes)

print("one-hot encoded")

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=.1, random_state=42)

print("XY train / test created")

images = None
labels = None

print("freeing some memory")

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjust the number of neurons to match the number of classes
])

print("model created")

# Compiling the model
model.compile(  optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("model compiled")


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('./../../models/model-zoomed.h5', monitor='val_accuracy', save_best_only=True)

print("callbacks implemented")

# Training the model
history = model.fit(
    X_train, y_train,
    batch_size=8,
    epochs=5,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint]
)

# Evaluating the model on the test set
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {val_accuracy * 100:.2f}%')
