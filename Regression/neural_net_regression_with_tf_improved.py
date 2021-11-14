
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Creating features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Turn our numpy arrays into tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

# Visualize it
# plt.scatter(X, y)
# plt.show()


# set random seed

tf.random.set_seed(42)



# Improve model
# 1. Creating the model
# More layers, increase the number of hidden units (neurons)
# change the activation function for each layer
# 2. Compile
# change the optimization function, learning rate
# 3. Fit the model
# number of epochs, more data

# 1. Create a model using the sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=None,  input_shape=(1,)),
    tf.keras.layers.Dense(100, activation=None),

    tf.keras.layers.Dense(1)
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

# 3. Fit the model
model.fit(X, y, epochs=95)

# check out X and y
print(X)
print(y)

# Make a prediction
print(model.predict([17.0]))









