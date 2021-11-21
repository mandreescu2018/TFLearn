# NN for classification
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

# check out features
print(X)

# check out labels
print(y[:10])

# visualize data
circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
print(circles)

# plot data
plt.scatter(X[:,0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# heck the shape of our features and labels
print(X.shape, y.shape)
print(X[0], y[0])

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1.fit(X, y, epochs=100)
model_1.evaluate(X, y)
predictions=model_1.predict(X)
print(predictions)

# Plot decision boundary
graphutils.plot_decision_boundary(model_1, X, y)


