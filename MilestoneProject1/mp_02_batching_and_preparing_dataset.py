import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import preprocessutils

datasets_list = tfds.list_builders()  # get all datasets available in TFDS
print("food101" in datasets_list)

# Load in the data food101
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,  # data gets returned in tuple format (data, label)
                                             with_info=True)

# Map preprocessing function to training data (and parallelization)
train_data = train_data.map(map_func=preprocessutils.preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map preprocessing function to test data
test_data = test_data.map(map_func=preprocessutils.preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

print(train_data)
print(test_data)

