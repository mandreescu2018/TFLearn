import tensorflow as tf
from tensorflow.keras import preprocessing, models
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

loaded_model = models.load_model("101_food_classes_10_percent_saved_big_dog_model")

res = loaded_model.evaluate(test_data)
print(res)
