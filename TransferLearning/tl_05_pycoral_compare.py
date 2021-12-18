import tensorflow as tf
from tensorflow.keras import applications, layers, Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Utils import transferlearningutils, graphutils

base_model = applications.MobileNetV2(weights='imagenet')
base_model.summary()

