import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random

# the data has already been sorted
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# show the first training example
# print(f"Training example:\n{train_data[0]}\n")
# print(f"Training label:\n{train_labels[0]}\n")
# hack shapes
print(train_data[0].shape)
plt.imshow(train_data[7])
# plt.show()

class_names =["T-shirt/top", "Trauser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot an example image and its label
index_of_choice = 151
plt.imshow(train_data[index_of_choice], cmap=plt.cm.binary)
plt.title(class_names[train_labels[index_of_choice]])
plt.show()

plt.figure(figsize=(7,7))
for i in range(4):
    ax = plt.subplot(2,2,i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])

plt.show()

