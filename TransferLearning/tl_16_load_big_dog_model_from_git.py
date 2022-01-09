import tensorflow as tf
from scipy.stats import yeojohnson_llf
from tensorflow.keras import preprocessing, models
from Utils import preprocessutils, graphutils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint

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

# print(test_data)

start = time.perf_counter()

y_labels = []
for images, labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())

# print(y_labels[:10])
#
# exit(0)

model = models.load_model("06_101_food_class_10_percent_saved_big_dog_model")

# res_downloaded_model = model.evaluate(test_data)
# print(res_downloaded_model)
# exit(0)

class_names = preprocessutils.get_classes("101_food_classes_10_percent/train")
# print("class_names: ", class_names)
# print('test_data.class_names: ', test_data.class_names)

preds_probs = model.predict(test_data, verbose=1)
print(len(preds_probs))
print(preds_probs.shape)
print(preds_probs[0], len(preds_probs[0]), max(preds_probs[0]))

end = time.perf_counter()
print(f"Inference time: {end - start} sec")

print(f"Class with highest probability: {tf.argmax(preds_probs[0])}")
print(f"Class name: {test_data.class_names[tf.argmax(preds_probs[0])]}")

pred_classes = preds_probs.argmax(axis=1)
print(pred_classes[:10])

sklearn_acc = accuracy_score(y_labels, pred_classes)
print(sklearn_acc)

# # make confusion matrix
# cm = confusion_matrix(y_true=y_labels, y_pred=pred_classes)
#
#
# graphutils.plot_confusion_matrix(cm, classes=test_data.class_names, figsize=(100, 100))

classif_report = classification_report(y_true=y_labels, y_pred=pred_classes, output_dict=True)

pprint(classif_report)

# Create empty dictionary
class_f1_scores = dict()

for k, v in classif_report.items():
    if k == "accuracy":
        break
    else:
        class_f1_scores[class_names[int(k)]] = v["f1-score"]

print("\n========== F1 scores =============")
# pprint(class_f1_scores)
f1_scores = pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)

print(f1_scores[:10])

fig, ax = plt.subplots(figsize=(12, 25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(f1_scores["class_names"])
ax.set_xlabel("F1-score")
ax.set_title("F1-scores for 101 Food classes")
plt.show()
