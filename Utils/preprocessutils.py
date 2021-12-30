import tensorflow as tf
import os


def load_and_preprocess_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


def walk_through_dir(pathname):
    for dirpath, dirnames, filenames in os.walk(pathname):
        print(f"There are {len(dirnames)} folders and {len(filenames)} images in '{dirpath}'")


if __name__ == '__main__':
    load_and_preprocess_image("/home/mihai/learn/TFLearn/ComputerVision/03-steak.jpg")
