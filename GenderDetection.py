from typing import Literal

import tensorflow as tf
import cv2

from keras.utils import image_dataset_from_directory
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Rescaling, RandomZoom, \
    RandomFlip, RandomRotation
from keras.metrics import SparseCategoricalAccuracy, Accuracy
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import SparseCategoricalCrossentropy

# from torch.nn import Conv2d, MaxPool2d, Flatten, Dropout, BatchNorm1d, Sequential, Linear, ReLU, CrossEntropyLoss
# from torch.optim import Adam, SGD, RMSprop
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

# %%

image_size = (256, 256)
batch_size = 32
validation_split = 0.15

parent_path = r"S:\Dataset\Computer Vision Project\Gender Detection Dataset\Dataset"

train, validation = image_dataset_from_directory(parent_path, image_size=image_size, batch_size=batch_size,
                                                 validation_split=validation_split, subset='both', seed=69)

class_names = train.class_names


# %%

def visualize_classes(which_split: Literal['train', 'validation'] = 'train') -> None:
    __data__ = (train, validation)[which_split == 'validation']
    __classes__ = __data__.class_names
    plt.figure(figsize=(10, 10))
    for __images, __labels in __data__.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(__images[i].numpy().astype('uint16'))
            plt.title(__classes__[__labels[i]])
            plt.axis('off')
            plt.tight_layout()
            plt.suptitle(f'Gender from {which_split}')

    plt.show()


visualize_classes()


# %%

def data_info(which_split: Literal['train', 'validation'] = 'train') -> None:
    __data__ = (train, validation)[which_split == 'validation']
    __classes__ = __data__.class_names

    for __image__ in __data__.take(1):
        __image__ = __image__[0].numpy()
        print(
            f"{which_split}\nsize: {len(__data__)}\nimage shape: {__image__.shape}\ndtype: {__image__.dtype}\nclasses: {class_names}\nclasses count: {len(class_names)}")


data_info('train')

# %%

data_augmentation = Sequential([
    RandomFlip('horizontal', input_shape=image_size + (3,)),
    RandomZoom(0.1),
    RandomRotation(0.1)
])

model = Sequential([
    data_augmentation,

    Rescaling(1.0 / 255),

    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(class_names)),
])

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy()]
)

history = model.fit(train, validation_data=validation, epochs=3, batch_size=32, workers=-1, use_multiprocessing=True,
                    validation_batch_size=16)

history = history.history
# %%
model.save('gender_detection_90plus')
