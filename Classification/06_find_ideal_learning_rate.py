import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utils import graphutils


# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# create training and test set
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])

# Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

history = model.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])

d_frame = pd.DataFrame(history.history)
print(d_frame)
d_frame.plot(figsize=(10, 7), xlabel="epochs")
plt.show()

# Plot the learning rate vs loss
lrs = 1e-4 * (10**(tf.range(100)/20))
print(lrs)

plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("learning rate")
plt.ylabel("loss")
plt.title("learning rate vs loss")
plt.show()

# appears that ideal learning rate is '0.02'
