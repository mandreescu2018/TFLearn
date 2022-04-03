import tensorflow as tf
import os
import pathlib
import numpy as np


def preprocess_img(image, label, img_shape=224):
    """
    Converts image data type from 'uint8' to 'float32' and reshape image
    :param image:
    :param label:
    :param img_shape:
    :return:
    """
    image = tf.image.resize(image, size=[img_shape, img_shape])
    # image = image/255.  # scale image values - For EfficientNetBx is not necessary
    return tf.cast(image, tf.float32), label


def load_and_preprocess_image(filename, img_shape=224):
    """
    Reads an image, turns into a tensor

    :param filename:
    :param img_shape:
    :param scale:
    :return:
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads an image, turns into a tensor

    :param filename:
    :param img_shape:
    :param scale:
    :return:
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    if scale:
        img = img / 255.
    return img


def walk_through_dir(pathname):
    for dirpath, dirnames, filenames in os.walk(pathname):
        print(f"There are {len(dirnames)} folders and {len(filenames)} images in '{dirpath}'")


def get_classes(path_to_train):
    # get the class names programmatically
    data_dir = pathlib.Path(path_to_train)
    return np.array(sorted([item.name for item in data_dir.glob("*")]))


if __name__ == '__main__':
    load_and_preprocess_image("/home/mihai/learn/TFLearn/ComputerVision/03-steak.jpg")
