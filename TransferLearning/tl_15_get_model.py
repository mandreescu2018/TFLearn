import wget
import zipfile

url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/06_101_food_class_10_percent_saved_big_dog_model.zip'
wget.download(url)

zip_ref = zipfile.ZipFile("06_101_food_class_10_percent_saved_big_dog_model.zip")
zip_ref.extractall()
zip_ref.close()
