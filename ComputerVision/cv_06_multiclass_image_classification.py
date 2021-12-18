import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
import cv_01_computer_vision_dataset as ds
import matplotlib.pyplot as plt
from Utils import graphutils, metricsutils
import wget

# img_url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-hamburger.jpeg'
# wget.download(img_url)
# exit(0)

model_name = "CNN_Multi2.h5"

tf.random.set_seed(42)

class_names = ds.get_food_classes()
train_dir = ds.get_food_data_dir()
test_dir = ds.get_food_data_dir(train=False)

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)

# Load data and turn into batches
# train_data = train_datagen.flow_from_directory(train_dir,
#                                                target_size=(224,224),
#                                                batch_size=32,
#                                                class_mode="categorical")

# Augmented data
train_data = train_datagen_augmented.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode="categorical")



# Create a model - start a baseline
model = tf.keras.Sequential([
    Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=(224,224,3)),
    # Activation(activation='relu'),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Conv2D(10,3, activation='relu'),
    Conv2D(10,3, activation='relu'), #simplify in order to reduce overfitting
    MaxPool2D(),
    Flatten(),
    Dense(10, activation='softmax')

])

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

if not os.path.isfile(model_name):
    start = time.perf_counter()
    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data)
              )
    end = time.perf_counter()
    print(f"Training duration: {end - start} sec")
    graphutils.plot_loss_curves(history)
    plt.show()
    model.save(model_name)
else:
    model = tf.keras.models.load_model(model_name)


# res_eval = model.evaluate(test_data)
# print(res_eval)

print(class_names)
# picture = "03-pizza-dad.jpeg"
picture = "03-hamburger.jpeg"
# picture = "03-sushi.jpeg"

metricsutils.pred_and_plot(model, picture, class_names=class_names)

