import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model

from sklearn.metrics import confusion_matrix
from Utils import graphutils
import os.path
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# the data has already been sorted
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# show the first training example
# print(f"Training example:\n{train_data[0]}\n")
# print(f"Training label:\n{train_labels[0]}\n")
# hack shapes
print(train_data[0].shape)

class_names =["T-shirt/top", "Trauser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# input shape: 28x28
# ooutpu shape: 10 len(class_names)
# loss CategoricalCrossEntropy if labels are one hot encoded, otherwise sparse CategoricalCrossEntropy
# output activation: softmax

train_data_norm = train_data/255.0
test_data_norm = test_data/255.0

print(train_data_norm.min(), train_data_norm.max())

if not os.path.isfile('my_model.h5'):
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

    # Create the learning rate callback
    lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

    history = model.fit(train_data_norm,
                        train_labels,
                        epochs=20,
                        validation_data=(test_data_norm, test_labels))

    model.save('my_model.h5')
else:
    model = tf.keras.models.load_model('my_model.h5')

print(model.layers[1])

# get the pattern of a layer in our network
weights, biases = model.layers[2].get_weights()

print(weights, weights.shape)
print(biases, biases.shape)

plot_model(model, show_shapes=True)