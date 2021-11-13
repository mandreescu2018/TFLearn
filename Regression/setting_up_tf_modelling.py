import tensorflow as tf
from Utils import graphutils, metricsutils
import pandas as pd

# 1. get more data
# 2. Make the model larger (more layers or more hidden units per layer)
# 3. Train for longer

# Make a dataset
X = tf.range(-100, 100, 4)
print(X)

# Make labels
y = X + 10
print(y)

# Split the data in train and test sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

print("X_test", X_test)
print("y_test", y_test)

# graphutils.data_visualizing(X_train, y_train, X_test, y_test)

# 3 modelling experiments
# - model_1 => 1 layer, 100 epochs
# - model_2 => 2 layers, 100 epochs
# - model_3 => 2 layers, 500 epochs

# Model 1
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])
model_1.summary()
model_1.fit(X_train, y_train, epochs=100)

y_preds_1 = model_1.predict(X_test)
# y_preds_1 = tf.squeeze(y_preds_1)
graphutils.plot_predictions(X_train, y_train, X_test, y_test, y_preds_1)

mae_1 = metricsutils.mae(y_true=y_test, y_pred=y_preds_1)
mse_1 = metricsutils.mse(y_true=y_test, y_pred=y_preds_1)
print("mae 1:", mae_1)
print("mse 1:", mse_1)

# Model 2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_2.summary()
model_2.fit(X_train, y_train, epochs=100)

y_preds_2 = model_2.predict(X_test)
graphutils.plot_predictions(X_train, y_train, X_test, y_test, y_preds_2)

mae_2 = metricsutils.mae(y_true=y_test, y_pred=y_preds_2)
mse_2 = metricsutils.mse(y_true=y_test, y_pred=y_preds_2)
print("mae 2:", mae_2)
print("mse 2:", mse_2)

# Model 3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_3.summary()
model_3.fit(X_train, y_train, epochs=500)

y_preds_3 = model_3.predict(X_test)
graphutils.plot_predictions(X_train, y_train, X_test, y_test, y_preds_3)

mae_3 = metricsutils.mae(y_true=y_test, y_pred=y_preds_3)
mse_3 = metricsutils.mse(y_true=y_test, y_pred=y_preds_3)
print("mae 3:", mae_3)
print("mse 3:", mse_3)

# compare using pandas
model_results =[["model_1", mae_1.numpy(), mse_1.numpy()],
                ["model_2", mae_2.numpy(), mse_2.numpy()],
                ["model_3", mae_3.numpy(), mse_3.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results)
