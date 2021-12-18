import os
import time

import tensorflow as tf
# Create the data loaders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from Utils import transferlearningutils, graphutils

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32
train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

train_data_10_percent = image_dataset_from_directory(directory=train_dir,
                                                     image_size=IMAGE_SIZE,
                                                     label_mode="categorical",
                                                     batch_size=BATCH_SIZE)

test_data_10_percent = image_dataset_from_directory(directory=test_dir,
                                                     image_size=IMAGE_SIZE,
                                                     label_mode="categorical",
                                                     batch_size=BATCH_SIZE)

print(train_data_10_percent)
print(train_data_10_percent.class_names)