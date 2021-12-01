import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
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


tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(#loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
model.summary()
non_norm_history = model.fit(train_data, tf.one_hot(train_labels, depth=10),
                             epochs=10, validation_data=(test_data, tf.one_hot(test_labels, depth=10)))



history = model.fit(train_data_norm, tf.one_hot(train_labels, depth=10), epochs=10,
                    validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)))



# plot non normalized data
pd.DataFrame(non_norm_history.history).plot(title="Non normalized")
pd.DataFrame(history.history).plot(title="Normalized")
plt.show()