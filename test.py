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


# read data

# generate training, validation and test data

# model
# vgg16
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


# test
def THREED_CNN(weights_path=None):
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
def VGG2D(x,d0,d1):

# 3d vgg



# cnn + lstm

if __name__ == "__main__":
    # summery test
    model1 = TEST_MODEL()
    model1.summary()
