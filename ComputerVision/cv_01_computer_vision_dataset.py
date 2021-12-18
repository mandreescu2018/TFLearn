# import tensorflow as tf
import zipfile
import wget
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# FOOD 101 DATASET

# GET THE DATA BINARY
# img_url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip'
# wget.download(img_url)

# UNZIP THE DATA

# zip_ref = zipfile.ZipFile("pizza_steak.zip")
# zip_ref.extractall()
# zip_ref.close()

# GET THE DATA MULTICLASS
# img_url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip'
# wget.download(img_url)
#
# # UNZIP THE DATA
#
# zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip")
# zip_ref.extractall()
# zip_ref.close()

# Inspect the data - Become one with the data
# walk through pizza_stea dir and list number of files


def view_random_image(target_dir, target_class):
    # setup the target directory
    target_folder = target_dir +"/" + target_class

    random_image = random.sample(os.listdir(target_folder), 1)

    img = mpimg.imread(target_folder +"/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape: {img.shape}")

    # plt.show()
    return img


def get_food_classes():
    # get the class names programmatically
    data_dir = pathlib.Path("10_food_classes_all_data/train")
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    class_names = class_names[1:]
    return class_names

def get_food_data_dir(train=True):
    if train:
        return "10_food_classes_all_data/train"
    return "10_food_classes_all_data/test"

def get_classes():
    # get the class names programmatically
    data_dir = pathlib.Path("pizza_steak/train")
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    class_names = class_names[1:]
    return class_names

def get_data_dir(train=True):
    if train:
        return "pizza_steak/train"
    return "pizza_steak/test"

if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
        print(f"There are {len(dirnames)} folders and {len(filenames)} images in '{dirpath}'")

    num_steak_images_train = len(os.listdir("10_food_classes_all_data/train/"))
    print("num_steak_images_train ",num_steak_images_train)

    print("classes: ", get_food_classes())

    img = view_random_image(target_dir="10_food_classes_all_data/train", target_class=random.choice(get_food_classes()))
    plt.show()
    # print(img, img.shape)
    # print(img/255.)


