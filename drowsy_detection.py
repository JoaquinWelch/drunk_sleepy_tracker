#buddha baker first

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import random
import shutil

# Image generator
train_data1 = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def class_split(src_folder, dest_train, dest_valid, split_ratio=0.8):
    classes = os.listdir(src_folder)
    
    for class_name in classes:
        class_folder = os.path.join(src_folder, class_name)
        images = os.listdir(class_folder)
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        training_images = images[:split_index]
        validation_images = images[split_index:]

        training_class_fold = os.path.join(dest_train, class_name)
        validation_class_fold = os.path.join(dest_valid, class_name)

        os.makedirs(training_class_fold, exist_ok=True)
        os.makedirs(validation_class_fold, exist_ok=True)

        for image in training_images:
            source_path = os.path.join(class_folder, image)
            destination_path = os.path.join(training_class_fold, image)
            shutil.copy(source_path, destination_path)

        for image in validation_images:
            source_path = os.path.join(class_folder, image)
            destination_path = os.path.join(validation_class_fold, image)
            shutil.copy(source_path, destination_path)

source_data_folder = r'C:\Users\Buck Bigwheat\DDD dataset'
proj_folder = r'C:\Users\Buck Bigwheat\drunk-tracker'
dest_train = r'C:\Users\Buck Bigwheat\DDD training-valid\training'
dest_valid = r'C:\Users\Buck Bigwheat\DDD training-valid\validation'

class_split(os.path.join(proj_folder, source_data_folder), dest_train, dest_valid, split_ratio=0.8)

training_wheels = train_data1.flow_from_directory(os.path.join(dest_train),
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  subset='training')

valid_generator = train_data1.flow_from_directory(os.path.join(dest_valid),
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  subset='validation')


import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=r'C:\Users\Buck Bigwheat\DDD training-valid\training', target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=r'C:\Users\Buck Bigwheat\DDD training-valid\validation', target_size=(224,224))

VGG = VGG16(input_shape=(244,244,3), include_top=False, weights='imagenet')

VGG.trainable = False

model = Sequential([
    VGG,
    Flatten(),
    Dense(units=256, activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=2, activation="softmax")
])
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata, validation_steps=10, epochs=5)
model.save('vggclf')

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Accuracy and Loss")
plt.ylabel("Value")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss"])
plt.show()
