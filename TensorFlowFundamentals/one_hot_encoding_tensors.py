import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# create a list of indices
some_list = [0, 1, 2, 3] # could be red, green, blue, purple

# One hot encode our list of indices
print(tf.one_hot(some_list, depth=4))

# specify some custom value for one-hot encoding
print(tf.one_hot(some_list, depth=4, on_value="a", off_value="b"))



