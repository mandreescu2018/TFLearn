import tensorflow as tf
from scipy.stats import yeojohnson_llf
from tensorflow.keras import preprocessing, models
from Utils import preprocessutils, graphutils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint

IMG_SIZE = (224, 224)

train_dir_101_classes = "101_food_classes_10_percent/train"
test_dir = "101_food_classes_10_percent/test/"

test_data = preprocessing.image_dataset_from_directory(test_dir,
                                                       label_mode="categorical",
                                                       image_size=IMG_SIZE,
                                                       shuffle=False)

y_labels = []
for images, labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())

class_names = preprocessutils.get_classes(train_dir_101_classes)

model = models.load_model("06_101_food_class_10_percent_saved_big_dog_model")

# graphutils.plot_predicted_images_random(test_dir, model, class_names)

# Finding the most wrong prediction

filepaths = []
for filepath in test_data.list_files(test_dir + "*/*.jpg", shuffle=False):
    filepaths.append(filepath.numpy())


# print(filepaths[:10])

preds_probs = model.predict(test_data, verbose=1)
pred_classes = preds_probs.argmax(axis=1)

# create a DataFrame of different parameters for each test img
pred_df = pd.DataFrame({"img_pat": filepaths,
                        "y_true": y_labels,
                        "y_pred": pred_classes,
                        "pred_conf": preds_probs.max(axis=1),
                        "y_true_classname": [class_names[i] for i in y_labels],
                        "y_pred_classname": [class_names[i] for i in pred_classes]})
pprint(pred_df[:10])

pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]


