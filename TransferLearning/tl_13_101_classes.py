import tensorflow as tf
from tensorflow.keras import preprocessing
from general_transferlearning import NeuralModel
from Utils import graphutils

IMG_SIZE = (224, 224)

train_dir_101_classes = "101_food_classes_10_percent/train"
test_dir = "101_food_classes_10_percent/test"

train_data_101_classes = preprocessing.image_dataset_from_directory(train_dir_101_classes,
                                                                    label_mode="categorical",
                                                                    image_size=IMG_SIZE)

test_data = preprocessing.image_dataset_from_directory(test_dir,
                                                       label_mode="categorical",
                                                       image_size=IMG_SIZE,
                                                       shuffle=False)

model_object_5 = NeuralModel(number_of_classes=len(train_data_101_classes.class_names),
                             save_best_only=True,
                             monitor='val_accuracy')
# model_5 = NeuralModel101()
# model_object_5.model.summary()
model_object_5.compile_model()
model_object_5.create_checkpoint_callback('101_classes_10_percent_data_model-checkpoint/checkpoint.ckpt')

history = model_object_5.model.fit(train_data_101_classes,
                                   epochs=5,
                                   validation_data=test_data,
                                   validation_steps=int(0.15 * len(test_data)),
                                   callbacks=[model_object_5.checkpoint_callback]
                                   )
res = model_object_5.model.evaluate(test_data)
print(res)

# graphutils.plot_loss_curves(history)

model_object_5.unfreeze_layers(no_of_layers=5)

fine_tune_epochs = 10

history_fine_tuning = model_object_5.model.fit(train_data_101_classes,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data,
                                               validation_steps=int(0.15 * len(test_data)),
                                               initial_epoch=history.epoch[-1])

res = model_object_5.model.evaluate(test_data)
print(res)
graphutils.compare_histories(history, history_fine_tuning)

model_object_5.model.save("101_food_classes_10_percent_saved_big_dog_model")
