import tensorflow as tf


def load_and_preprocess_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img/255.
    return img


if __name__ == '__main__':
    load_and_preprocess_image("/home/mihai/learn/TFLearn/ComputerVision/03-steak.jpg")
