import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, callbacks


class NeuralModel():
    def __init__(self, number_of_classes=10,
                 save_best_only=False,
                 monitor='val_loss'):
        self.base_model = applications.EfficientNetB0(include_top=False)
        self.base_model.trainable = False
        self.model = None
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.number_of_classes = number_of_classes
        self.checkpoint_callback = None
        self.create_model()

    def create_model(self):
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

        # Create the inputs and outputs
        inputs = layers.Input(shape=input_shape, name="input_layer")
        x = data_augmentation(inputs)  # augment training images
        x = self.base_model(x, training=False)  # pass augmented images to base model but keep in inference mode
        x = layers.GlobalAveragePooling2D(name="Global_average_pooling")(x)
        outputs = layers.Dense(self.number_of_classes, activation="softmax", name="output_layer")(x)
        model = keras.Model(inputs, outputs)
        self.model = model

    def save_model(self, model_name):
        self.model.save(model_name)

    def compile_model(self, learning_rate=0.001):
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizers.Adam(learning_rate=learning_rate),
                           metrics=['accuracy'])

    def create_checkpoint_callback(self, checkpoint_path):
        self.checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=self.save_best_only,
                                                             monitor=self.monitor,
                                                             save_freq="epoch",
                                                             verbose=1)

    def unfreeze_layers(self, no_of_layers=10):
        # unfreeze all the layers in base model
        self.base_model.trainable = True

        # Refreeze all layers except last 'no_of_layers'
        for layer in self.base_model.layers[:-no_of_layers]:
            layer.trainable = False

        self.compile_model(learning_rate=0.0001)




