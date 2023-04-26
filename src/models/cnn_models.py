import tensorflow as tf
from keras import layers 
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten, Dense, Dropout, Input, UpSampling1D, UpSampling2D, concatenate, TimeDistributed, Conv1D, Reshape, Concatenate

import numpy as np


# 1D CNN architecture
def build_1d_cnn(input_shape, num_classes=4, dropout_rate=0.5):
    inputs = Input(input_shape)

    conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    flatten = Flatten()(pool2)
    dense = Dense(128, activation='relu')(flatten)

    outputs = Dense(num_classes * input_shape[0], activation='softmax')(dense)
    outputs = Reshape((input_shape[0], num_classes))(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 2D CNN architecture 
def build_2d_cnn(input_shape, num_classes=4, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Upsampling path
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))

    # Final output layer
    model.add(Conv2D(num_classes, kernel_size=(1, 1), activation='softmax'))

    return model

# 1D CNN + 2D CNN U-Net architecture
def build_1d_2d_cnn(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Process the spectral information using a 1D CNN
    spectral_conv = TimeDistributed(Conv1D(32, 3, activation='relu', padding='same'))(inputs)
    reshaped_spectral_conv = Reshape((input_shape[0], input_shape[1], 32))(spectral_conv)

    # Process the spatial information using a 2D U-Net
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped_spectral_conv)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    up4 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)

    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 2D CNN + 1D CNN architecture
def build_2d_1d_cnn(input_shape, num_classes):
    inputs = Input(input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    reshaped = Reshape((-1, input_shape[-1]))(pool2)

    conv3 = Conv1D(128, 3, activation='relu', padding='same')(reshaped)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    flatten = Flatten()(pool3)
    dense = Dense(256, activation='relu')(flatten)

    outputs = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 3D CNN architecture
def build_3d_cnn(input_shape, num_classes):
    inputs = Input(input_shape)

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    flatten = Flatten()(pool3)
    dense = Dense(256, activation='relu')(flatten)

    outputs = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# ResNet50 architecture
def build_resnet50(input_shape, num_classes):
    pass  # Define your ResNet50 architecture here


# def build_3d_cnn_model(input_shape):
#     model = Sequential()

#     # Add 3D convolutional layers
#     model.add(layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

#     model.add(layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
#     model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

#     # Add fully connected layers
#     model.add(layers.Flatten())
#     model.add(layers.Dense(units=128, activation='relu'))
#     model.add(layers.Dense(units=64, activation='relu'))

#     # Add the output layer with the number of classes
#     num_classes = 2  # Update the number of classes based on your dataset
#     model.add(layers.Dense(units=num_classes, activation='softmax'))

#     # Compile the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     return 

