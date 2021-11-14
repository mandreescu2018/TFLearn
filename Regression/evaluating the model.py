# Evaluating the model
# ! Visualize, Visualize, Visualize !
# - The data
# - The model
# - The training of a model
# - The predictions

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from Utils import metricsutils, graphutils
import numpy as np
import matplotlib.pyplot as plt

# Make a bigger dataset
X = tf.range(-100, 100, 4)
print(X)

# Make labels
y = X + 10
print(y)

# plt.scatter(X, y)
# plt.show()

#  The 3 sets
# The training set (70-80%) - training
# The validation set (10-15%) - tuning
# The test set (10-15%) -evaluation

# check the length
print(len(X))

# Split the data in train and test sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

print("X_test", X_test)

tf.random.set_seed(42)

# 1. create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# 2. Compile
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()
#
from tensorflow.keras.utils import plot_model
#
plot_model(model, show_shapes=True)

# 3. Fit
model.fit(X_train, y_train, epochs=100)

# Make some predictions
y_pred = model.predict(X_test)
# y_pred = [row[2] for row in y_pred]
print(y_pred)

graphutils.plot_predictions(train_data=X_train, train_labels=y_train,
                            test_data=X_test, test_labels=y_test, predictions=y_pred)

# Evaluate the model om the test set
model.evaluate(X_test, y_test)
err_ae = tf.reduce_mean(y_test - y_pred)
print("mean abs err:", err_ae)
err_ae = metricsutils.mae(y_test, tf.squeeze(tf.constant(y_pred)))
print("mean abs err:", err_ae)

# Calculate the mean squared error
err_ms = metricsutils.mse(y_test, tf.squeeze(y_pred))
print("squared err: ", err_ms)