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

print(ds_info.features)

# Get the class names
class_names = ds_info.features["label"].names
print(class_names[:10])

# Exploring
# one sample of the train data
train_one_sample = train_data.take(1)
print(train_one_sample)

for image, label in train_one_sample:
    print(f"Image shape: {image.shape},\n "
          f"Image datatype: {image.dtype}, \n"
          f"Target class from Food101 (tensor form) {label}, \n"
          f"Class name (string form): {class_names[label.numpy()]}")
    # print(image)
    print(tf.reduce_min(image), tf.reduce_max(image))
    plt.imshow(image)
    plt.title(class_names[label.numpy()])
    plt.axis(False)
    plt.show()
    preprocessed_img = preprocessutils.preprocess_img(image, label)[0]
    print("Image before preprocessing: ", image[:2])
    print("shape", image.shape)
    print("\nImage after preprocessing ")
    print(preprocessed_img[:2])
    print("shape", preprocessed_img.shape)



