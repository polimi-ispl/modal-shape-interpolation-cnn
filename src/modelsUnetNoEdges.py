"""
Architectures for mode shape images reconstruction
"""

import keras
from keras import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization, UpSampling2D, MaxPooling2D

interp_noedges = Input((6, 10, 1))


# The proposed architecture MaxReNet

x11 = Convolution2D(32, (3, 3), input_shape=(6, 10, 1), padding='same')(interp_noedges)
x11 = Activation('relu')(x11)
x11 = BatchNormalization()(x11)
x11 = MaxPooling2D((2, 2))(x11)
intds11 = MaxPooling2D((2, 2))(interp_noedges)
stack11 = keras.layers.concatenate([x11, intds11])

x21 = Convolution2D(64, (3, 3), padding='same')(stack11)
x21 = Activation('relu')(x21)
x21 = BatchNormalization()(x21)
x21 = MaxPooling2D((2, 2), strides=(1, 1))(x21)
intds21 = MaxPooling2D((2, 2), strides=(1, 1))(intds11)
stack21 = keras.layers.concatenate([x21, intds21])

x31 = Convolution2D(128, (3, 3), padding='same')(stack21)
x31 = Activation('relu')(x31)
x31 = BatchNormalization()(x31)
x31 = MaxPooling2D((2, 2))(x31)
intds31 = MaxPooling2D((2, 2))(intds21)
stack31 = keras.layers.concatenate([x31, intds31])

y21 = UpSampling2D((2, 2))(stack31)
y21 = Convolution2D(64, (3, 3), padding='same')(y21)
y21 = Activation('relu')(y21)
y21 = BatchNormalization()(y21)
y21 = keras.layers.add([x21, y21])

y11 = UpSampling2D((2, 2))(y21)
y11 = Convolution2D(32, (2, 4), padding='valid')(y11)
y11 = Activation('relu')(y11)
y11 = BatchNormalization()(y11)
y11 = keras.layers.add([x11, y11])

yb = UpSampling2D((2, 2))(y11)
yb = Convolution2D(1, (3, 3), padding='same')(yb)
yb = Activation('relu')(yb)
yb = BatchNormalization()(yb)
yb = keras.layers.add([interp_noedges, yb])

uNet5Stackb = Model(interp_noedges, yb)


# Simpler autoencoder with no MaxRe connections (unet)

x11c = Convolution2D(32, (3, 3), input_shape=(6, 10, 1), padding='same')(interp_noedges)
x11c = Activation('relu')(x11c)
x11c = BatchNormalization()(x11c)
x11c = MaxPooling2D((2, 2))(x11c)

x21c = Convolution2D(64, (3, 3), padding='same')(x11c)
x21c = Activation('relu')(x21c)
x21c = BatchNormalization()(x21c)
x21c = MaxPooling2D((2, 2), strides=(1, 1))(x21c)


x31c = Convolution2D(128, (3, 3), padding='same')(x21c)
x31c = Activation('relu')(x31c)
x31c = BatchNormalization()(x31c)
x31c = MaxPooling2D((2, 2))(x31c)

y21c = UpSampling2D((2, 2))(x31c)
y21c = Convolution2D(64, (3, 3), padding='same')(y21c)
y21c = Activation('relu')(y21c)
y21c = BatchNormalization()(y21c)
y21c = keras.layers.add([x21c, y21c])

y11c = UpSampling2D((2, 2))(y21c)
y11c = Convolution2D(32, (2, 4), padding='valid')(y11c)
y11c = Activation('relu')(y11c)
y11c = BatchNormalization()(y11c)
y11c = keras.layers.add([x11c, y11c])

ybc = UpSampling2D((2, 2))(y11c)
ybc = Convolution2D(1, (3, 3), padding='same')(ybc)
ybc = Activation('relu')(ybc)
ybc = BatchNormalization()(ybc)
ybc = keras.layers.add([interp_noedges, ybc])

uNet5f2c = Model(interp_noedges, ybc)
