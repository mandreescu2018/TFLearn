import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import cv_01_computer_vision_dataset as ds
from Utils import graphutils

tf.random.set_seed(42)

model_name = "CNN1.h5"
# Visualize the data
# plt.figure()
# plt.subplot(1, 2, 1)
# steak_img = ds.view_random_image("pizza_steak/train/", "steak")
# plt.subplot(1, 2, 2)
# pizza_img = ds.view_random_image("pizza_steak/train/", "pizza")
# plt.show()


train_dir = ds.get_data_dir()
test_dir = ds.get_data_dir(train=False)

# Turn data in batches
# Create train and test data generators and rescale the data
train_datagen = ImageDataGenerator(rescale=1 / 255.)
test_datagen = ImageDataGenerator(rescale=1 / 255.)

train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.3,
                                             horizontal_flip=True)

# Load our image data from dirs and turn them into batches
train_data = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                         target_size=(224, 224),
                                                         class_mode="binary",
                                                         batch_size=32,
                                                         shuffle=True)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(224, 224),
                                             class_mode="binary",
                                             batch_size=32)

images, labels = train_data.next()
print(len(images), len(labels))
print(labels)

# Induce overfitting:
#   - increase the number of conv layer
#   - increase the number of conv filter
#   - add another dense layer to output of flatten layer
# Reduce overfitting:
#   - Add data augmentation
#   - Add regularization layers (such as MaxPool2D)
#   - More data

# Create the model (baseline)
model = Sequential([
    Conv2D(filters=10, kernel_size=3, strides=1, padding="valid", activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

# Fit the model
model.summary()

if not os.path.isfile(model_name):
    start = time.perf_counter()
    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

    end = time.perf_counter()
    print(f"Training Duration: {end - start} sec")

else:
    model = tf.keras.models.load_model(model_name)

# pd.DataFrame(history.history).plot(figsize=(10, 7))
# When validation loss start to increase, the model is overfitting

graphutils.plot_loss_curves(history)
plt.show()
