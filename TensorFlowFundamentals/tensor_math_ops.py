import os

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Create a new tensor
H = tf.range(1, 10)
print(H)
print(tf.square(H))

# find square root
print("=====square root===\n")
print(tf.sqrt(tf.cast(H, dtype=tf.float32)))

# find the Log
print("=====Log===\n")
print(tf.math.log(tf.cast(H, dtype=tf.float32)))



