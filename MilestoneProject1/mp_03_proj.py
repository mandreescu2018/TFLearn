import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import mixed_precision, layers, applications, optimizers, Model
from tensorflow.keras.layers.experimental import preprocessing

mixed_precision.set_global_policy("mixed_float16")


import matplotlib.pyplot as plt
from Utils import preprocessutils, transferlearningutils

datasets_list = tfds.list_builders()  # get all datasets available in TFDS
print("food101" in datasets_list)

# Load in the data food101
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,  # data gets returned in tuple format (data, label)
                                             with_info=True)

class_names = ds_info.features["label"].names


# Map preprocessing function to training data (and parallelization)
train_data = train_data.map(map_func=preprocessutils.preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=100).batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map preprocessing function to test data
test_data = test_data.map(map_func=preprocessutils.preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

print(train_data)
print(test_data)

# create ModelCheckpoint callback

checkpoint_path = "model_checkpoints/cp.ckpt"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor="val_acc",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=0)

print("mixed_precision.global_policy", mixed_precision.global_policy())

# Build feature extraction model

input_shape = (224, 224, 3)
base_model = applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create functional model
inputs = layers.Input(shape=input_shape, name="input_layer", dtype=tf.float16)
# Note: EfficientNet has rescaling built-in,
# but if you have another layer you can use:
# x = preprocessing.Rescaling(1./255)(x)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(len(class_names))(x)
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizers.Adam(),
              metrics=["accuracy"])

# model.summary()
# print("\nCheck for model")
# for layer in model.layers:
#     print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
#
# # Check dtype policy for our base model
# print("\nCheck for base model")
# for layer in model.layers[1].layers[:20]:
#     print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Fit the feature extraction model

history101 = model.fit(train_data,
                       epochs=3,
                       steps_per_epoch=len(train_data),
                       validation_data=test_data,
                       batch_size=16,
                       validation_steps=int(0.15 * len(test_data)),
                       callbacks=[transferlearningutils.create_tensorboard_callback(dir_name="training_logs",
                                                                                    experiment_name="efficientnetb0_101"),
                                  model_checkpoint])

res = model.evaluate()
print(res)

