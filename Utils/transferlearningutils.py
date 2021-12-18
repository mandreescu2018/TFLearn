import datetime
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_model(model_url, num_classes=10, image_shape=(224, 224)):
    """

    :param model_url:
    :param num_classes:
    :return: An uncompiled model with model_url as feature extractor
            and Dense output layer with 'num_classes' output neurons

    """
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False, # Freeze already learned patterns
                                             name="feature_extraction_layer",
                                             input_shape=image_shape+(3,))

    # Create model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model




