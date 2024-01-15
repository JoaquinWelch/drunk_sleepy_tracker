#buddha baker first

from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import shutil

# Image generator
train_data1 = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def class_split(src_folder, proj_folder, split_ratio=0.8):
    classes = os.listdir(src_folder)
    
    for class_name in classes:
        class_folder = os.path.join(src_folder, class_name)
        images = os.listdir(class_folder)
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        training_images = images[:split_index]
        validation_images = images[split_index:]




        training_class_fold = os.path.join(proj_folder, 'training', class_name)
        os.makedirs(training_class_fold, exist_ok=True)
        os.makedirs(validation_class_fold, exist_ok=True)
        validation_class_fold = os.path.join(proj_folder, 'validation', class_name)





        for image in training_images:
            source_path = os.path.join(class_folder, image)
            destination_path = os.path.join(training_class_fold, image)
            shutil.copy(source_path, destination_path)

        for image in validation_images:
            source_path = os.path.join(class_folder, image)
            destination_path = os.path.join(validation_class_fold, image)
            shutil.copy(source_path, destination_path)