from Utils import preprocessutils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import time
from Utils import transferlearningutils, graphutils


# create dirs
train_dir_1_percent = "10_food_classes_1_percent/train"
test_dir_1_percent = "10_food_classes_1_percent/test"
test_dir_all = "../ComputerVision/10_food_classes_all_data/test"

# How many images we have
# preprocessutils.walk_through_dir('10_food_classes_1_percent')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
train_data_1_percent = keras.preprocessing.image_dataset_from_directory(
    train_dir_1_percent,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_data_1_percent = keras.preprocessing.image_dataset_from_directory(
    test_dir_1_percent,
    label_mode="categorical",
    image_size=IMG_SIZE
)

test_data = keras.preprocessing.image_dataset_from_directory(
    test_dir_all,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# preprocessutils.walk_through_dir("10_food_classes_1_percent")

# Adding data augmentation right into the model
# (preprocessing is on GPU - much faster)

# layers.experimental.preprocessing()

# Create data augmentation stage: flipping, rotation, ...

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomHeight(0.2),
    layers.experimental.preprocessing.RandomWidth(0.2),
    # layers.experimental.preprocessing.Rescaling(1./255)
    ], name='data_augmentation'
)

# ====== TEST & VIEW AUGMENTED PICS =================

# print(train_data_1_percent.class_names)
# target_class = random.choice(train_data_1_percent.class_names)
# target_dir = train_dir_1_percent + "/" + target_class
# print(target_dir)
# random_image = random.choice(os.listdir(target_dir))
# random_image_path = target_dir + "/" + random_image


# img = mpimg.imread(random_image_path)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Original")
# augmented_image = data_augmentation(img)
# plt.subplot(1, 2, 2)
# plt.imshow(augmented_image)
# plt.title("Augmented")
# plt.show()
# ===========================================

# setup input shape and base model, freeze the base model layer
input_shape = (224, 224, 3)

base_model = applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentation sequential
x = data_augmentation(inputs)

# give base_model the inputs layer
x = base_model(x, training=False)

# Pool output features of the base_model
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

# Put a dense layer as the output
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

# Make a model using the inputs and outputs
model_1 = keras.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",
                optimizer=optimizers.Adam(),
                metrics=["accuracy"])

start = time.perf_counter()

# Fit the model
history_1_percent = model_1.fit(train_data_1_percent,
                                epochs=5,
                                steps_per_epoch=len(train_data_1_percent),
                                validation_data=test_data_1_percent,
                                validation_steps=int(0.25 * len(test_data_1_percent)),
                                # track the training
                                callbacks=[transferlearningutils.create_tensorboard_callback(dir_name="transfer_learning",
                                                                                             experiment_name="1_percent_data")]
                                )

end = time.perf_counter()
print(f"Training duration: {end-start} sec")

graphutils.plot_loss_curves(history_1_percent)
