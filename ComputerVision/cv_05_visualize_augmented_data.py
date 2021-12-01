from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv_01_computer_vision_dataset as ds
import random
import matplotlib.pyplot as plt

train_dir = ds.get_data_dir()
test_dir = ds.get_data_dir(train=False)

IMG_SIZE = (224, 224)

print("Augmented training data:")
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)


train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2, # rotate an image with 0.2
                                             shear_range=0.2, # decupare 0.2
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)


train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=IMG_SIZE,
                                                                   batch_size=32,
                                                                   class_mode="binary",
                                                                   shuffle=False)

print("Non augmented training data:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               target_size=IMG_SIZE,
                                               class_mode="binary",
                                               shuffle=False)


# Visualize some augmented data
images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next()

# show original image and augmented image
random_number = random.randint(0,32)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(images[random_number])
plt.title("Original")
plt.axis(False)
plt.subplot(1,2,2)
plt.imshow(augmented_images[random_number])
plt.title("augmented")
plt.axis(False)
plt.show()
