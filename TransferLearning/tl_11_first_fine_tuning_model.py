# FINE TUNING AN EXISTING MODEL ON 10% OF DATA

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, callbacks
from Utils import transferlearningutils, graphutils
from general_transferlearning import NeuralModel

# create dirs
train_dir_10_percent = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
initial_epochs = 5

train_data_10_percent = keras.preprocessing.image_dataset_from_directory(
    train_dir_10_percent,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_data = keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE
)

model_2 = NeuralModel()


model_2.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=["accuracy"])

checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"
checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=False,
                                                save_freq="epoch",
                                                verbose=1)

start = time.perf_counter()

history_10_percent_ata_aug = model_2.model.fit(train_data_10_percent,
                                               epochs=initial_epochs,
                                               steps_per_epoch=len(train_data_10_percent),
                                               validation_data=test_data,
                                               validation_steps=int(0.25 * len(test_data)),
                                               callbacks=[transferlearningutils.create_tensorboard_callback(
                                                   dir_name="transfer_learning_10",
                                                   experiment_name="10_percent"),
                                                   checkpoint_callback]
                                               )
end = time.perf_counter()

print(f"Training first part duration: {end - start} sec")

model_2.model.load_weights(checkpoint_path)

# How many trainable variables are in our base model
print(len(model_2.model.layers[2].trainable_variables))

# Set last 10 layers trainable
model_2.base_model.trainable = True

# Freeze all layers except last 10
for layer in model_2.base_model.layers[:-10]:
    layer.trainable = False

# Recompile model after every change
model_2.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(0.0001),  # lower learning rate for tuning
                      metrics=["accuracy"])

print(len(model_2.model.layers[2].trainable_variables))

# Fine tune for another 5 epochs
fine_tune_epochs = initial_epochs + 5

start = time.perf_counter()

# Refit the model
history_fine_tuning = model_2.model.fit(train_data_10_percent,
                                        epochs=fine_tune_epochs,
                                        validation_data=test_data,
                                        validation_steps=int(0.25 * len(test_data)),
                                        initial_epoch=4,
                                        callbacks=[transferlearningutils.create_tensorboard_callback(
                                            dir_name="transfer_learning_10",
                                            experiment_name="10_percent_fine_tuned"),
                                            checkpoint_callback]
                                        )

end = time.perf_counter()
print(f"Training second part duration: {end - start} sec")

res = model_2.model.evaluate(test_data)
print(res)
# graphutils.plot_loss_curves(history_fine_tuning)
graphutils.compare_histories(history_10_percent_ata_aug, history_fine_tuning)
