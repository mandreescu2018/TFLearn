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

# Create a model checkpoint callback that saves the model's weights only
checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=False,
                                                save_freq="epoch",
                                                verbose=1)

start = time.perf_counter()

initial_epochs = 5
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
print(f"Training duration: {end-start} sec")
print(history_10_percent_ata_aug.epoch[-1])
res = model_2.model.evaluate(test_data)
print(res)
graphutils.plot_loss_curves(history_10_percent_ata_aug)