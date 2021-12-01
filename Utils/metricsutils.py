import tensorflow as tf
from Utils import preprocessutils
import matplotlib.pyplot as plt

def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, tf.squeeze(y_pred))


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, tf.squeeze(y_pred))

def pred_and_plot(model, filename, class_names=None):
    img = preprocessutils.load_and_preprocess_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[int(tf.round(pred))]

    plt.imshow(img)
    plt.title(f'Prediction: {pred_class}')
    plt.axis(False)
    plt.show()
