import gc
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
# from prettytable import PrettyTable
from IPython.display import Image

# from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten


# read data

# generate training, validation and test data

# model
# test
def TEST_MODEL(weights_path=None):
    model = tf.keras.Sequential()
    model.add(Input(shape=(100,)))
    model.add(Embedding(21, 64, input_length=100))
    model.add(
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
    model.add(Dense(1000))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

# 2d vgg
def vgg2d():
    model = Sequential()
    model.add(Conv2D(64, (3, 3),
           activation='relu',
           padding='same',
           name='block1_conv1',
           dim_ordering='tf',
           input_shape=(255,255,3)))
    model.add(Conv2D(64, (3, 3),
           activation='relu',
           padding='same',
           name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),
           activation='relu',
           padding='same',
           name='block2_conv2'))
    print(model.output.shape)
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    model.add(Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1'))
    model.add(Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2'))
    model.add(Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1'))
    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2'))
    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1'))
    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2'))

    model.add(Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu',name='fc1'))

    model.add(Dense(4096, activation='relu', name='fc2'))

    model.add(Dense(2, activation='sigmoid', name='predictions'))
    print(model.summary())
    return model

# 3d vgg
def vgg3d():
    model = Sequential()
    model.add(Conv3D(64, (3, 3,3),
           activation='relu',
           padding='same',
           name='block1_conv1',
           dim_ordering='tf',
           # 64x64x33 cube
           input_shape=(64,64,33,1)))
    model.add(Conv3D(64, (3, 3,3),
           activation='relu',
           padding='same',
           name='block1_conv2'))
    model.add(MaxPooling3D(
            pool_size=(2, 2,2),
            strides=(2, 2,2),
            name='block1_pool',
            padding='same'
    ))
    model.add(Conv3D(128, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1'))
    model.add(Conv3D(128, (3, 3,3),
           activation='relu',
           padding='same',
           name='block2_conv2'))

    model.add(MaxPooling3D((2,2, 2), strides=(2,2, 2),#padding='same',
                           name='block2_pool'
                           ))

    model.add(Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1'))
    model.add(Conv3D(256, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2'))
    model.add(Conv3D(256, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3'))
    model.add(MaxPooling3D((2,2, 2), strides=(2,2, 2),padding='same', name='block3_pool'))
    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1'))
    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2'))
    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3'))
    model.add(MaxPooling3D((2,2, 2), strides=(2,2, 2),padding='same', name='block4_pool'))
    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1'))
    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2'))

    model.add(Conv3D(512, (3,3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3'))
    model.add(MaxPooling3D((2,2, 2), strides=(2,2, 2),padding='same', name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu',name='fc1'))

    model.add(Dense(4096, activation='relu',name='fc2'))

    model.add(Dense(2, activation='sigmoid', name='predictions'))
    print(model.summary())
    return model


# cnn + lstm

if __name__ == "__main__":
    # summery test
    vgg3d()
    # model1 = TEST_MODEL()
    # model1.summary()
