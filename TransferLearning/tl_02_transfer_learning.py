import os
import time

# Create the data loaders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Utils import transferlearningutils, graphutils

# dependencies for pretrained
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import models
import matplotlib.pyplot as plt


IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

model_name = "CNN_Transfer.h5"
model_name_2 = "RSNET.h5"

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training images:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=IMAGE_SHAPE,
                                               batch_size=BATCH_SIZE,
                                               class_mode="categorical")

print("Testing images:")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=IMAGE_SHAPE,
                                             batch_size=BATCH_SIZE,
                                             class_mode="categorical")



efficientnet_url = 'https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1'
resnet_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'

# create ResNet model
resnet_model = transferlearningutils.create_model(resnet_url,
                                                  num_classes=train_data.num_classes,
                                                  image_shape=IMAGE_SHAPE)
resnet_model.summary()

resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])

if not os.path.isfile(model_name):

    start = time.perf_counter()

    history = resnet_model.fit(train_data,
                                 epochs=5,
                                 batch_size=BATCH_SIZE,
                                 steps_per_epoch=len(train_data),
                                 validation_data=test_data,
                                 validation_steps=len(test_data),
                                 callbacks=[transferlearningutils.create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                              experiment_name="resnet50V2"
                                                                                              )]
                                 )

    end = time.perf_counter()
    print(f"Training duration: {end - start} sec")

    resnet_model.save(model_name)
# else:
#     resnet_model = models.load_model(model_name)
#
# if history:
#     graphutils.plot_loss_curves(history)
#     plt.show()

# resnet_model.evaluate(test_data)

efficinetnet_model = transferlearningutils.create_model(efficientnet_url,
                                                            num_classes=train_data.num_classes,
                                                            image_shape=IMAGE_SHAPE)
efficinetnet_model.summary()
print(efficinetnet_model.layers[0].weights)

if not os.path.isfile(model_name_2):


    efficinetnet_model.compile(loss="categorical_crossentropy",
                               optimizer=tf.keras.optimizers.Adam(),
                               metrics=["accuracy"])
    start = time.perf_counter()
    history_ef = efficinetnet_model.fit(train_data,
                                        epochs=5,
                                        batch_size=BATCH_SIZE,
                                        steps_per_epoch=len(train_data),
                                        validation_data=test_data,
                                        validation_steps=len(test_data),
                                        callbacks=[transferlearningutils.create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                                     experiment_name="efficientnetb0")]
                                        )
    end = time.perf_counter()
    print(f"Training duration: {end - start} sec")

    efficinetnet_model.save(model_name_2)
else:
    exit(0)
    efficientnet_url = models.load_model(model_name_2)

if history_ef:
    graphutils.plot_loss_curves(history_ef)
    plt.show()