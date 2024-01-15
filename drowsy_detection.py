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