# Introduction to regression
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

# Creating features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
print(X + 10)
# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
# plt.scatter(X, y)
# plt.show()

# # Input and output shapes
# Create a demo tensor for our prediction problem
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
print(house_info)
print(house_price)

input_shape = X[0].shape
output_shape = y[0].shape
print("input_shape", input_shape)
print("output_shape", output_shape)

# Turn our numpy arrays into tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)
print(X.shape)
print(y)

input_shape = X[0].shape
output_shape = y[0].shape
print("input_shape", input_shape)
print("output_shape", output_shape)

## Modelling with tensorflow

# set random seed
tf.random.set_seed(42)

# 1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# 3. Fit the model
model.fit(X, y, epochs=35)

# check out X and y
print(X)
print(y)

# Make a prediction
print(model.predict([17.0]))








