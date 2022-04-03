import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utils import graphutils
from sklearn.metrics import confusion_matrix


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

# se the learning rate 0.02
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=20)

y_preds = model.predict(X_test)
# print(y_preds[:10], tf.round(y_preds[:10]))
model_loss, model_accuracy = model.evaluate(X_test, y_test)
print("loss: ", model_loss, "accuracy:", model_accuracy)

cm = confusion_matrix(y_true=y_test, y_pred=tf.round(y_preds))
print(cm)

graphutils.plot_confusion_matrix(cm)
# plt.show()


