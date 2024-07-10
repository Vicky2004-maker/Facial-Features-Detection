from typing import Literal, Optional
import os

import cv2
import tensorflow as tf

from keras.models import load_model
from keras.utils import image_dataset_from_directory, load_img, img_to_array
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Rescaling, RandomZoom, \
    RandomFlip, RandomRotation
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

import matplotlib.pyplot as plt
import numpy as np


class GenderDetection:
    def __init__(self):
        self.history = None
        self.model = None
        self.data_augmentation = None
        self.model_path = "models/gender_detection_90plus.keras"

        self.image_size = (256, 256)
        self.batch_size = 32
        self.validation_split = 0.15

        self.parent_path = r"S:\Dataset\Computer Vision Project\Gender Detection Dataset\Dataset"

        self.train, self.validation = image_dataset_from_directory(self.parent_path, image_size=self.image_size,
                                                                   batch_size=self.batch_size,
                                                                   validation_split=self.validation_split,
                                                                   subset='both',
                                                                   seed=69)

        self.class_names = self.train.class_names

        self.load()

    def load(self):
        self.model = Sequential()
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=True)
        else:
            self.data_augmentation = Sequential([
                RandomFlip('horizontal', input_shape=self.image_size + (3,)),
                RandomZoom(0.1),
                RandomRotation(0.1)
            ])

            self.model = Sequential([
                self.data_augmentation,

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

                Dropout(0.15),

                Flatten(),

                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(4, activation='relu'),
                Dense(len(self.class_names)),
            ])

            self.model.compile(
                optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy()]
            )

            history = self.model.fit(self.train, validation_data=self.validation, epochs=2, batch_size=32, workers=-1,
                                     use_multiprocessing=True,
                                     validation_batch_size=self.batch_size)

            self.history = history.history
            self.model.save(self.model_path)

    def visualize_classes(self, which_split: Literal['train', 'validation'] = 'train', img_count: int = 9) -> None:
        __data__ = (self.train, self.validation)[which_split == 'validation']
        __classes__ = __data__.class_names
        plt.figure(figsize=(10, 10))
        for __images, __labels in __data__.take(1):
            for i in range(img_count):
                plt.subplot(int(np.sqrt(img_count)), int(np.sqrt(img_count)), i + 1)
                plt.imshow(__images[i].numpy().astype('uint16'))
                plt.title(__classes__[__labels[i]])
                plt.axis('off')
                plt.tight_layout()
                plt.suptitle(f'Gender from {which_split}')

        plt.show()

    def data_info(self, which_split: Literal['train', 'validation'] = 'train') -> None:
        __data__ = (self.train, self.validation)[which_split == 'validation']
        __classes__ = __data__.class_names

        for __image__ in __data__.take(1):
            __image__ = __image__[0].numpy()
            print(
                f"{which_split}\nsize: {len(__data__)}\nimage shape: {__image__.shape}\ndtype: {__image__.dtype}\nclasses: {self.class_names}\nclasses count: {len(self.class_names)}")

    def predict_numpy(self, img_arr: np.ndarray, cv: bool = True, print_output: bool = True) -> Optional[str]:
        if cv:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img_arr, self.image_size)
        print(img_arr.shape)
        img_arr = tf.expand_dims(img_arr, 0)
        predictions = tf.nn.softmax(self.model.predict(img_arr))
        if print_output:
            print(
                f'Predicted the image as {self.class_names[np.argmax(predictions)]} with {np.max(predictions) * 100:.2f}%')
            return None
        else:
            return self.class_names[np.argmax(predictions)]

    def test_with_image(self, image_path: str, visualize: bool = True):
        test_img = load_img(image_path, target_size=self.image_size)
        test_img = img_to_array(test_img)
        test_img = tf.expand_dims(test_img, 0)
        predictions = tf.nn.softmax(self.model.predict(test_img))

        if visualize:
            plt.imshow(load_img(image_path, target_size=self.image_size))
            plt.axis('off')
            plt.suptitle('Prediction')
            plt.title(
                f'Image Path: {image_path}\nPredicted: {self.class_names[np.argmax(predictions)]} ({np.max(predictions) * 100:.2f}%)')
            plt.tight_layout()
            plt.show()

        else:
            print(
                f'Predicted the image located at {image_path} as {self.class_names[np.argmax(predictions)]} with {np.max(predictions) * 100:.2f}%')

    def get_model(self) -> Sequential:
        return self.model

    def get_model_path(self) -> str:
        return self.model_path
