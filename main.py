import numpy as np
from pathlib import Path
import cv2
#import sys
import glob
import os
import argparse
import seaborn as sns

from keras import models
from keras import layers
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
#from keras.utils import to_categorical
#from keras.metrics import Precision, Recall, AUC
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


"""
Global parameters:
image_size: loaded images are resized to that size
penalty: penalty in l2 regularization
drop_rate: parameter for Dropout layers, 0.1 means 10% of neurons outputs are made 0.
"""
image_size = 64
penalty = 0.1
drop_rate = 0.33

def load_train_data(path):
    """
    Loads training data from given directory

    param path: path to the directory where training data are kept
    """
    folder_path = Path(path)
    images = np.zeros((2062, image_size, image_size, 1), dtype="float32")
    classes = np.zeros((2062))
    i = 0

    for j in range(0, 10):
        filepath = folder_path / str(j)
        jpg_files = glob.glob(os.path.join(filepath, "*.JPG"))
        for jpg in jpg_files:
            image = cv2.cvtColor(np.array(cv2.imread(str(jpg))), cv2.COLOR_BGR2GRAY)
            print(jpg)
            if image is not None:
                image = cv2.resize(image, (image_size, image_size))
                images[i, :, :, 0] = image
                classes[i] = j
                i+=1
    images = images / 255.0
    print(f"Loaded {len(images)} training images and their corresponding classes.")
    return images, classes

def load_test_data(path):
    """
    Loads test data from given directory

    param path: path to the directory where test data are kept
    """
    filepath = Path(path).expanduser()
    jpg_files = sorted(list(filepath.rglob("*.JPG")))
    print(f"jpg files: {jpg_files}")
    test_images = np.zeros((10, image_size, image_size, 1), dtype="float32")
    test_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    i = 0
    for jpg in jpg_files:
        image = cv2.cvtColor(np.array(cv2.imread(str(jpg))), cv2.COLOR_BGR2GRAY)
        if image is not None:
            image = cv2.resize(image, (image_size, image_size))
            test_images[i, :, :, 0] = image
            i+=1
    print(f"Loaded {len(test_images)} test images and their corresponding classes.")
    test_images = test_images / 255.0
    return test_images, test_classes

def neural_network(images, classes, test_images, test_classes):
    """
    Creates and runs neural_network using keras library
    based on Conv2D and MaxPool2D layers

    param images: training images
    param classes: training classes
    param test_images: test images
    param test_classes: test classes

    """
    print("Creating neural network.")
    network = models.Sequential()
    network.add(layers.Conv2D(60, (5, 5), input_shape=(image_size, image_size, 1), kernel_regularizer=l2(penalty)))
    network.add(BatchNormalization())
    network.add(layers.Activation('relu'))
    network.add(layers.MaxPool2D((3, 3)))
    network.add(layers.Dropout(drop_rate))
    # old layers, commented because they leaded to overtraining
    """
    network.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2(penalty)))
    network.add(BatchNormalization())
    network.add(layers.Activation('relu'))
    network.add(layers.MaxPool2D((2, 2)))
    network.add(layers.Dropout(drop_rate))
    network.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(penalty)))
    network.add(BatchNormalization())
    network.add(layers.Activation('relu'))
    network.add(layers.MaxPool2D((2, 2)))
    network.add(layers.Dropout(drop_rate))
    network.add(layers.Flatten())
    network.add(layers.Dense(32, kernel_regularizer=l2(penalty)))
    network.add(BatchNormalization())
    network.add(layers.Activation('relu'))
    """
    network.add(layers.Flatten())
    network.add(layers.Dense(10, activation="softmax"))

    optimizer = RMSprop()
    network.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("Neural network compile successfully.")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
    images, val_images, classes, val_classes = train_test_split(images, classes, test_size=0.2)
    history = network.fit(images, classes, batch_size=200, epochs=50, validation_data=(val_images, val_classes), callbacks = [early_stopping])
    print("Training of neural network finished successfully.")

    test_loss, test_acc = network.evaluate(test_images, test_classes)
    print("Testing of neural network ended successfully.")
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc} \n")

    predictions = network.predict(test_images)
    print("Predictions:")
    print(predictions)

    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted classes:")
    print(predicted_classes)

    plot_accuracy(history)

    network.save('trained_model.h5')
    print("Model saved to 'trained_model.h5'")

    return predictions, predicted_classes

def plot_accuracy(history):
    """
    Plots the accuracy of the neural network

    param history: fitted neural network
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test a neural network.")
    parser.add_argument('--train-data-path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--test-data-path', type=str, required=True, help='Path to the testing data.')
    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    images, classes = load_train_data(train_data_path)
    test_images, test_classes = load_test_data(test_data_path)
    predictions, predicted_classes = neural_network(images, classes, test_images, test_classes)
