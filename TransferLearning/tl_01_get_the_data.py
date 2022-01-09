import wget
import zipfile
import os

# GET THE DATA MULTICLASS
# img_url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip'
# img_url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip'
# wget.download(img_url)

# # UNZIP THE DATA
#
# zip_ref = zipfile.ZipFile("101_food_classes_10_percent.zip")
# zip_ref.extractall()
# zip_ref.close()

for dirpath, dirnames, filenames in os.walk("101_food_classes_10_percent"):
    print(f"There are {len(dirnames)} diractories and {len(filenames)} images in '{dirpath}'")



