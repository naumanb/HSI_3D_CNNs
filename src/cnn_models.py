import tensorflow as tf
from keras import layers 
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Dense, Dropout, Input, UpSampling2D, concatenate

import numpy as np

# 2D CNN architecture 
# Input shape should be (height, width, num_spectra, 1)
def build_2d_cnn(input_shape, num_classes=4, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# 3D CNN architecture
# Input shape should be (num_images, height, width, num_spectra, 1)
def build_3d_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#  U-Net architecture 
def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv5)

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

