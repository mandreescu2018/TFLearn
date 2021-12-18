import tensorflow as tf
from tensorflow.keras import applications, layers, Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Utils import transferlearningutils, graphutils

import tl_03_transfer_learning_fine_tune as tl

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

# 1. Create base model with keras.applications
base_model = applications.EfficientNetB0(include_top=False)

# 2. Freeze the base model - keep weights already trained
base_model.trainable = False

# 3. Create inputs
inputs = layers.Input(shape=IMAGE_SIZE+(3,), name="input_layer")

# 4. (Optional) If using a model like ResNet50V2 you will need normalized inputs
# x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)

# 5. Pass the inputs to the base model
x = base_model(inputs)
print(f"Shape after passing inputs to the model: {x.shape}")

# 6. Average pool the outputs
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"Shape after GlobalAveragePooling2D: {x.shape}")

# 7. Create the output activation layer
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs in a model
model = Model(inputs, outputs)

# check the layers

for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name)
base_model.summary()
exit(0)
# ============

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(tl.train_data_10_percent,
          epochs=10,
          batch_size=BATCH_SIZE,
          steps_per_epoch=len(tl.train_data_10_percent),
          validation_data=tl.test_data_10_percent,
          validation_steps=int(0.25*len(tl.test_data_10_percent)),
          callbacks=[transferlearningutils.create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                 experiment_name="transfer_learning")]
                    )

graphutils.plot_loss_curves(history)


