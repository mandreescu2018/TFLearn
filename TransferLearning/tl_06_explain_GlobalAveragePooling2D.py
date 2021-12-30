import tensorflow as tf
from tensorflow.keras import layers

input_shape = (1,4,4,3)
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print("input_tensor", input_tensor)

# FEATURE VECTOR

# pass the random tensor through a global average pooling 2D layer
global_average_pooled_tensor = layers.GlobalAveragePooling2D()(input_tensor)
global_max_pooled_tensor = layers.GlobalMaxPool2D()(input_tensor)

print("\nglobal_average_pooled_tensor", global_average_pooled_tensor)
print("\nglobal_average_pooled_tensor shape", global_average_pooled_tensor.shape)

print("\nglobal_max_pooled_tensor", global_max_pooled_tensor)
print("\nglobal_max_pooled_tensor shape", global_max_pooled_tensor.shape)

print("\n Replicate GlobalAveragePooling2D")
print(tf.reduce_mean(input_tensor, axis=[1,2]))


print("\n Replicate GlobalMaxPooling2D")
print(tf.reduce_max(input_tensor, axis=[1,2]))


