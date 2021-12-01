from sklearn.svm._liblinear import train_wrap

import cv_01_computer_vision_dataset as ds

import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPU time: Training Duration: 46.2313429 sec
# An end-to-end example

tf.random.set_seed(42)

# preprocess data - normalize
train_datagen = ImageDataGenerator(rescale=1/255.)
valid_datagen = ImageDataGenerator(rescale=1/255.)

train_dir = ds.get_data_dir()
test_dir = ds.get_data_dir(train=False)

# Import data and turn into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                              batch_size=32,
                                              target_size=(224, 224),
                                              class_mode="binary",
                                              seed=42)

print(len(train_data))

# Build a model like in classification,
# first with 4 neurons in Dense layers (2 Dense with 4 units)
# result: val_accuracy: 0.5040
# then try with 100 neurons, 3 Dense layers


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(units=100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()
# exit(0)



start = time.perf_counter()
history = model.fit(train_data,
                    epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data)
                    )
end = time.perf_counter()
print(f"Training Duration: {end - start} sec")

# loss, accuracy = model.evaluate()