# FINE TUNING AN EXISTING MODEL ON ALL OF DATA

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, callbacks
from Utils import transferlearningutils, graphutils
import general_transferlearning

train_dir_all = "../ComputerVision/10_food_classes_all_data/train"
test_dir_all = "../ComputerVision/10_food_classes_all_data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
initial_epochs = 5

train_data = keras.preprocessing.image_dataset_from_directory(
    train_dir_all,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_data = keras.preprocessing.image_dataset_from_directory(
    test_dir_all,
    label_mode="categorical",
    image_size=IMG_SIZE
)

model_4 = general_transferlearning.NeuralModel()

checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"
model_4.model.load_weights(checkpoint_path)

model_4.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(learning_rate=0.0001),
                      metrics=["accuracy"])

res = model_4.model.evaluate(test_data)
print(res)

# Set last 10 layers trainable
model_4.base_model.trainable = True

# Freeze all layers except last 10
for layer in model_4.base_model.layers[:-10]:
    layer.trainable = False

model_4.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(learning_rate=0.0001),
                      metrics=["accuracy"])

fine_tune_epochs = initial_epochs + 5

history_all_data = model_4.model.fit(train_data,
                                     epochs=fine_tune_epochs,
                                     steps_per_epoch=len(train_data),
                                     validation_data=test_data,
                                     validation_steps=int(0.25 * len(test_data)),
                                     initial_epoch=4,
                                     callbacks=[transferlearningutils.create_tensorboard_callback(
                                         dir_name="transfer_learning_hub",
                                         experiment_name="full_data")]
                                     )

res = model_4.model.evaluate(test_data)
print(res)
graphutils.plot_loss_curves(history_all_data)