from typing import Literal, Optional

import cv2
import keras
import tensorflow as tf

from keras.models import load_model
from keras.utils import image_dataset_from_directory, load_img, img_to_array
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Rescaling, RandomZoom, \
    RandomFlip, RandomRotation
from keras.metrics import SparseCategoricalAccuracy, Accuracy
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import SparseCategoricalCrossentropy

import matplotlib.pyplot as plt
import numpy as np

import os

parent_path = r"S:\Dataset\Computer Vision Project\Age Detection Dataset\data6"
model_path = "../models/age_classification_balanced.keras"
image_size = (256, 256)
batch_size = 64
validation_split = 0.15
seed = 69

train, validation = image_dataset_from_directory(parent_path, image_size=image_size,
                                                 batch_size=batch_size,
                                                 validation_split=validation_split,
                                                 subset='both',
                                                 seed=seed)

class_names = train.class_names

data_augmentation = Sequential([
    RandomFlip('horizontal', input_shape=image_size + (3,)),
    RandomZoom(0.1),
    RandomRotation(0.1)
])

model = Sequential([
    data_augmentation,

    Rescaling(1.0 / 255),

    Conv2D(4, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(8, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Dropout(0.1),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(len(class_names)),
])

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy()]
)

history = model.fit(train, validation_data=validation, epochs=10, batch_size=batch_size,
                    workers=-1,
                    use_multiprocessing=True,
                    validation_batch_size=batch_size)

history = history.history
