import tensorflow as tf


def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, tf.squeeze(y_pred))


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, tf.squeeze(y_pred))