import os

import tensorflow as tf
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# TensorFlow interacts beautifully with numpy

# Create a tensor directly from a numpy array
J = tf.constant(np.array([1., 7., 10.]))
print(J)

# convert to numpy array using numpy
np_array = np.array(J)
print(np_array, type(np_array))

# convert to numpy array using built in  numpy()
np_array = J.numpy()
print(np_array, type(np_array))

# the default types for each are slightly different
numpy_J = tf.constant(np.array([2., 7., 10.]))
tensor_J = tf.constant([2., 7., 10.])
print(numpy_J.dtype, tensor_J.dtype)

# check devices available
print('===== physical devices ======')
# print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices("GPU"))
