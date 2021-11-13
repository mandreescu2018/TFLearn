import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from Utils import metricsutils, graphutils
import numpy as np
import matplotlib.pyplot as plt

# Make a dataset
X = tf.range(-100, 100, 4)
# Make labels
y = X + 10

# Split the data in train and test sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

tf.random.set_seed(42)
# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model.summary()
model.fit(X_train, y_train, epochs=500)

model.save("best_model_SavedModel_format")
model.save("best_model_HDF5_format.h5")

saved_model = tf.keras.models.load_model("best_model_HDF5_format.h5")

y_preds = saved_model.predict(X_test)
graphutils.plot_predictions(X_train, y_train, X_test, y_test, y_preds)

mae = metricsutils.mae(y_true=y_test, y_pred=y_preds)
mse = metricsutils.mse(y_true=y_test, y_pred=y_preds)
print("mae :", mae)
print("mse :", mse)
