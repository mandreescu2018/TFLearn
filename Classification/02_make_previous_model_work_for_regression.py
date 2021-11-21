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
# plt.scatter(X[:,0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# heck the shape of our features and labels
print(X.shape, y.shape)
print(X[0], y[0])

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=(1,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])

# create some regression data
X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5)
print(X_regression)

# splt data
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]

y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]


# Fit model
model_1.fit(X_reg_train, y_reg_train, epochs=100)

# make pred
y_reg_preds = model_1.predict(X_reg_test)

# Plot preds
# graphutils.plot_decision_boundary(model_1, X_reg_test, y_reg_preds)
plt.figure(figsize=(10,7))
plt.scatter(X_reg_train, y_reg_train, c="b", label="Training data")
plt.scatter(X_reg_test, y_reg_test, c="g", label="Test data")
plt.scatter(X_reg_test, y_reg_preds, c="r", label="Pred data")
plt.legend()
plt.show()
