import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, callbacks
from Utils import transferlearningutils, graphutils

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

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomHeight(0.2),
    layers.experimental.preprocessing.RandomWidth(0.2)
    # layers.experimental.preprocessing.Rescaling(1./255) # for Efficient net... data scaling is already in the model
], name="data_augmentation"
)

# setup input shape and base model, freeze the base model layer
input_shape = (224, 224, 3)

base_model = applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create the inputs and outputs
inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)  # augment training images
x = base_model(x, training=False)  # pass augmented images to base model but keep in inference mode
x = layers.GlobalAveragePooling2D(name="Gglobal_average_pooling")(x)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
model_2 = keras.Model(inputs, outputs)

model_2.compile(loss='categorical_crossentropy',
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
history_10_percent_ata_aug = model_2.fit(train_data_10_percent,
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

graphutils.plot_loss_curves(history_10_percent_ata_aug)