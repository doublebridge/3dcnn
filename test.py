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
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv3D, MaxPooling3D, Flatten, TimeDistributed


import nibabel as nib
# import skimage.io as io

ospath = os.getcwd()

# read data
def load_data(partition):
    data = []
    for fn in os.listdir(os.path.join(ospath, 'data', partition)):
        img = nib.load(os.path.join(ospath,'data', partition, fn))
        datai = img.get_fdata()
       # img = np.array(img)
        data.append(datai)
    return data


def cnn3dlstm():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1',
                     dim_ordering='tf',
                     # 64x64x33 cube
                     input_shape=(64, 64, 33, 1)))
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling3D(
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        name='block1_pool',
        padding='same'
    ))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2),  # padding='same',
                           name='block2_pool'
                           ))

    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block3_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block4_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))

    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block5_pool'))

    model.add(Flatten(name='flatten'))

    lstm = Sequential()
    lstm.add(TimeDistributed(model))
    lstm.add(LSTM(100, input_shape=(100, 2)))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(100, activation="relu"))
    lstm.add(Dense(2, activation="softmax"))
    lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# THIS IS A BUG
    # lstm.build(input_shape=(100,2))

    return lstm
def vgg3d():
    model = Sequential()
    model.add(Conv3D(64, (3, 3,3),
           activation='relu',
           padding='same',
           name='block1_conv1',
           dim_ordering='tf',
           # 64x64x33 cube
           input_shape=(64,64,33,1)))
    model.add(Conv3D(64, (3, 3, 3),
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

    model.add(Dense(1, activation='sigmoid', name='predictions'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # print(model.summary())
    return model


if __name__ == "__main__":

    #load data
    train_x = np.array(load_data("spatial"))
    print(train_x.shape)
    train_y = np.ones(120)
    print(train_y)
    print(train_y.shape)

    #reshape data
    train_X=np.transpose(train_x, (0, 4, 1, 2, 3))
    train_X=np.reshape(train_X,(120,45,54,45))
    print(train_X.shape)

    train_X = pad_sequences(train_X, maxlen=33,dtype='int32',padding='post', value=0)
    train_X = np.pad(train_X,((0,0,),(0,0),(5,5),(9,10)),'constant')
    print(train_X.shape)#(xxx,33,64,64)
    train_X = np.transpose(train_X, (0,2,3,1))
    # (xxx,64, 64,33,1)
    train_X = train_X[:,:,:,:,np.newaxis]
    print(train_X.shape)


    model=vgg3d()
    history1= model.fit(train_X,train_y,epochs=20,batch_size=32)



