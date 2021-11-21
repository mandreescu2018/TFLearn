import tensorflow as tf
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Utils import graphutils

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

tf.random.set_seed(42)

# create training and test set
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

print(X_train.shape, X_test.shape)

# recreate the model
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), #increase the learning rate by 10 times (default=0.001)
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=25)

eval_res=model.evaluate(X_test, y_test)
print(eval_res)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Train")
graphutils.plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
graphutils.plot_decision_boundary(model, X_test, y_test)


plt.show()

# plot the loss
d_frame = pd.DataFrame(history.history)
print(d_frame)

d_frame.plot()
plt.title("loss curves")
plt.show()

# Find the best lerning rate
# - callback function
# - another model