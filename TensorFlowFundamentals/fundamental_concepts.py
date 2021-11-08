import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

scalar = tf.constant(7)
if __name__ == '__main__':
    print(scalar, "dimensions:",  scalar.ndim)

