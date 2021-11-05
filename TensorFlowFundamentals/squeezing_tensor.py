import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

tf.random.set_seed(42)
X = tf.constant(tf.random.uniform(shape=[50]), shape=(1,1,1,1,50))
print(X)

print(tf.squeeze(X))