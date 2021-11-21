import tensorflow as tf
import matplotlib.pyplot as plt

# Create a toy tensor
A = tf.cast(tf.range(-10, 10), dtype=tf.float32)
print(A)

# visualize the toy tensor
# plt.plot(A)
# plt.show()

# replicating sigmoid

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def relu(x):
    return tf.maximum(0, x)

# use the sigmoid function to our toy
res = sigmoid(A)
print(res)

plt.plot(sigmoid(A))
plt.show()

res = relu(A)
print(res)

plt.plot(relu(A))
plt.show()