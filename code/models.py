# -*- coding: utf-8 -*-

"""
This module contains the Tensorflow Keras models.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import tensorflow as tf
import wget
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, TimeDistributed


def _add_conv_block(num_filters, inputs, is_training, name_prefix, kernel_regularizer=None):
    """Adds a Batchnorm enabled convolution block.
    """
    conv = layers.Conv2D(num_filters,
                         (3, 3),
                         activation=None,
                         padding="same",
                         kernel_regularizer=kernel_regularizer,
                         name=name_prefix)(inputs)
    batchnorm = layers.BatchNormalization(name="%s-BN" % name_prefix)(conv, training=is_training)
    relu = layers.ReLU(name="%s-Relu" % name_prefix)(batchnorm)
    return relu


def _add_dense_block(units, inputs, is_training, name_prefix, kernel_regularizer=None, dropout_rate=0.5):
    """Adds a Batchnorm and Dropout enabled dense block.
    """
    dense = layers.Dense(units, activation=None, name=name_prefix)(inputs)
    batchnorm = layers.BatchNormalization(name="%s-BN" % name_prefix)(dense, training=is_training)
    relu = layers.ReLU(name="%s-Relu" % name_prefix)(batchnorm)
    dropout = layers.Dropout(dropout_rate, name="%s-Dropout_%s" % (name_prefix, dropout_rate))(relu, training=is_training)
    return dropout


def vgg16_batchnorm(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    """
    Builds a vgg16 model with Batchnorm and Dropout layers.

    :param input_shape: The shape of the input images (do not consider the batch-dimension).
    :param is_training: Flag to denote if the model would be used in training mode or not.
    :param num_classes: The number of classes. If it is binary classification, use num_classes=1.
    :return: The constructed tf.Keras model (Functional).

    """
    inputs = keras.Input(shape=input_shape, name='input')

    block1_conv1 = _add_conv_block(64, inputs, is_training, "block1_conv1")
    block1_conv2 = _add_conv_block(64, block1_conv1, is_training, "block1_conv2")
    block1_pool = layers.MaxPooling2D(name="block1_pool")(block1_conv2)

    block2_conv1 = _add_conv_block(128, block1_pool, is_training, "block2_conv1")
    block2_conv2 = _add_conv_block(128, block2_conv1, is_training, "block2_conv2")
    block2_pool = layers.MaxPooling2D(name="block2_pool")(block2_conv2)

    block3_conv1 = _add_conv_block(256, block2_pool, is_training, "block3_conv1")
    block3_conv2 = _add_conv_block(256, block3_conv1, is_training, "block3_conv2")
    block3_conv3 = _add_conv_block(256, block3_conv2, is_training, "block3_conv3")
    block3_pool = layers.MaxPooling2D(name="block3_pool")(block3_conv3)

    block4_conv1 = _add_conv_block(512, block3_pool, is_training, "block4_conv1")
    block4_conv2 = _add_conv_block(512, block4_conv1, is_training, "block4_conv2")
    block4_conv3 = _add_conv_block(512, block4_conv2, is_training, "block4_conv3")
    block4_pool = layers.MaxPooling2D(name="block4_pool")(block4_conv3)

    block5_conv1 = _add_conv_block(512, block4_pool, is_training, "block5_conv1")
    block5_conv2 = _add_conv_block(512, block5_conv1, is_training, "block5_conv2")
    block5_conv3 = _add_conv_block(512, block5_conv2, is_training, "block5_conv3")
    block5_pool = layers.MaxPooling2D(name="block5_pool")(block5_conv3)

    flatten = layers.Flatten(name="flatten")(block5_pool)
    fc1 = _add_dense_block(4096, flatten, is_training, "fc1")
    fc2 = _add_dense_block(4096, fc1, is_training, "fc2")

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(fc2)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(fc2)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="vgg16_batchnorm")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def vgg16_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
    output_vgg16_conv = model_vgg16_conv(inputs)

    # The training argument doesn't effect due to no Batchnorm and Dropout.
    flatten = layers.Flatten(name="flatten")(output_vgg16_conv)
    fc1 = _add_dense_block(4096, flatten, is_training, "fc1")
    fc2 = _add_dense_block(4096, fc1, is_training, "fc2")

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(fc2)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(fc2)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="vgg16_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet50_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet50_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet101_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet101(weights='imagenet', include_top=False)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet101_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_mask_pretrained_imagenet(input_shape, is_training=False, 
                                       num_classes=1, learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", "resnet152_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_pretrained_imagenet")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_mask_mog2_10channel_pretrained_imagenet(input_shape, 
                                                      is_training=False, 
                                                      num_classes=1, 
                                                      learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", 
                                "resnet152_10channel_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_10channel_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_pretrained_imagenet")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet152_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet152v2_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet152v2_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def inceptionresnetv2_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="inceptionresnetv2_pretrained_imagenet")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet50_pretrained_imagenet_lstm(input_shape, is_training=True, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    model_pretrained_conv_time_dist = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    flattened = TimeDistributed(layers.Flatten(name='flatten'))(model_pretrained_conv_time_dist)

    lstm_layer = LSTM(128, time_major=False)(flattened, training=is_training)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(lstm_layer)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet50_pretrained_imagenet_lstm")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet50_pretrained_imagenet_lstm_avg_pool(input_shape, is_training=False, num_classes=1, learning_rate=0.001):

    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    model_pretrained_conv_time_dist = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(layers.GlobalAveragePooling2D(name="avg_pool"))(model_pretrained_conv_time_dist)

    lstm_layer = LSTM(256, time_major=False)(avg_pool, training=is_training)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(lstm_layer)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet50_pretrained_imagenet_lstm_avg_pool")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_pretrained_imagenet_lstm_avg_pool(input_shape, is_training=False, num_classes=1, learning_rate=0.001):

    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)
    model_pretrained_conv_time_dist = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(layers.GlobalAveragePooling2D(name="avg_pool"))(model_pretrained_conv_time_dist)

    lstm_layer = LSTM(256, time_major=False)(avg_pool, training=is_training)

    if num_classes <= 2:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(lstm_layer)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet152_pretrained_imagenet_lstm_avg_pool")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model

def resnet50_pretrained_imagenet_lstm_custom_loss(input_shape, is_training=False, num_classes=1, learning_rate=0.001):

    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    model_pretrained_conv_time_dist = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(layers.GlobalAveragePooling2D(name="avg_pool"))(model_pretrained_conv_time_dist)

    lstm_layer = LSTM(256, time_major=False, return_sequences=True)(avg_pool, training=is_training)


    if num_classes <= 2:
        predictions = TimeDistributed(layers.Dense(1, activation="sigmoid", name="predictions"))(lstm_layer)
        predictions = layers.GlobalAveragePooling1D(name="1Dpool")(predictions)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet50_pretrained_imagenet_lstm_custom_loss")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model

def resnet152_pretrained_imagenet_lstm_custom_loss(input_shape, is_training=False, num_classes=1, learning_rate=0.001):

    inputs = keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)
    model_pretrained_conv_time_dist = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(layers.GlobalAveragePooling2D(name="avg_pool"))(model_pretrained_conv_time_dist)

    lstm_layer = LSTM(256, time_major=False, return_sequences=True)(avg_pool, training=is_training)


    if num_classes <= 2:
        predictions = TimeDistributed(layers.Dense(1, activation="sigmoid", name="predictions"))(lstm_layer)
        predictions = layers.GlobalAveragePooling1D(name="1Dpool")(predictions)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = keras.Model(inputs=inputs, outputs=predictions, name="resnet152_pretrained_imagenet_lstm_custom_loss")
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss=loss,
                  metrics=['accuracy', keras.metrics.AUC(curve='ROC')])

    return model

def resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_avg_pool(input_shape, 
                                                      is_training=False, 
                                                      num_classes=1, 
                                                      learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", 
                                "resnet152_10channel_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_10channel_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"))(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_avg_pool")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model

def resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_custom_loss(input_shape, 
                                                      is_training=False, 
                                                      num_classes=1, 
                                                      learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", 
                                "resnet152_10channel_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_10channel_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"))(output_pretrained_conv)

    lstm_layer = LSTM(256, time_major=False, return_sequences=True)(avg_pool, training=is_training)

    if num_classes <= 2:
        predictions = TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid", name="predictions"))(lstm_layer)
        predictions = layers.GlobalAveragePooling1D(name="1Dpool")(predictions)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_custom_loss")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model

def resnet152_mask_pretrained_imagenet_lstm_avg_pool(input_shape, is_training=False, 
                                       num_classes=1, learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", "resnet152_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"))(output_pretrained_conv)

    lstm_layer = LSTM(256, time_major=False)(avg_pool, training=is_training)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(lstm_layer)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_pretrained_imagenet_lstm_avg_pool")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model

def resnet152_mask_pretrained_imagenet_lstm_custom_pool(input_shape, is_training=False, 
                                       num_classes=1, learning_rate=0.001):
    
    weights_file = os.path.join(os.getcwd(), "..", "data", "resnet152_mask_weights.npy")
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/resnet152_mask_weights.npy"

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)
        
    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, 
                                                            include_top=False, 
                                                            input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = TimeDistributed(model_pretrained_conv)(inputs, training=is_training)

    avg_pool = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"))(output_pretrained_conv)

    lstm_layer = LSTM(256, time_major=False, return_sequences=True)(avg_pool, training=is_training)

    if num_classes <= 2:
        predictions = TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid", name="predictions"))(lstm_layer)
        predictions = layers.GlobalAveragePooling1D(name="1Dpool")(predictions)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(lstm_layer)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, 
                           name="resnet152_mask_pretrained_imagenet_lstm_custom_pool")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss=loss, 
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_6channel_opticalflow(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_6channel_opticalflow"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False,  input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_6channel_allpretrained_opticalflow(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_6channel_allpretrained_opticalflow"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False,  input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_15channel_opticalflow(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_15channel_opticalflow"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False,  input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_15channel_allpretrained_opticalflow(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_15channel_allpretrained_opticalflow"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False,  input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_13channel_allpretrained_hybrid(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_13channel_allpretrained_hybrid"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False, input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_16channel_allpretrained_hybrid(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_16channel_allpretrained_hybrid"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False, input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


def resnet152_7channel_allpretrained_maskopticalflow(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    # XXX: Lots of redundant  code.
    # This is the only line that changes.
    model_name = "resnet152_7channel_allpretrained_maskopticalflow"

    weights_file = os.path.join(os.getcwd(), "..", "data", "%s_weights.npy" % model_name)
    url = "https://capstonestorageaccount.blob.core.windows.net/capstone-container/%s" % os.path.basename(weights_file)

    # Check if the weights file is present else download it
    if not os.path.isfile(weights_file):
        wget.download(url, weights_file)

    # Load the weights
    weights = np.load(weights_file, allow_pickle=True)

    # Build the model
    inputs = tf.keras.Input(shape=input_shape, name='input')

    model_pretrained_conv = tf.keras.applications.ResNet152(weights=None, include_top=False,  input_shape=input_shape)
    model_pretrained_conv.set_weights(weights)
    output_pretrained_conv = model_pretrained_conv(inputs, training=is_training)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(output_pretrained_conv)

    if num_classes <= 2:
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(avg_pool)
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="predictions")(avg_pool)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # Note: one-hot labels are NOT required.

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    return model


# The dictionary mapping model names to model architecture functions.
# Ensure that the name of the model architecture matches with the model's 'name' attribute.
AVAILABLE_MODEL_ARCHS = {
    # Single-image models.
    "vgg16_batchnorm": vgg16_batchnorm,
    # Single-image models with pre-trained weights.
    "vgg16_pretrained_imagenet": vgg16_pretrained_imagenet,
    "resnet50_pretrained_imagenet": resnet50_pretrained_imagenet,
    "resnet101_pretrained_imagenet": resnet101_pretrained_imagenet,
    "resnet152_pretrained_imagenet": resnet152_pretrained_imagenet,
    "resnet152v2_pretrained_imagenet": resnet152v2_pretrained_imagenet,
    "inceptionresnetv2_pretrained_imagenet": inceptionresnetv2_pretrained_imagenet,

    # Sequence-based models.
    "resnet50_pretrained_imagenet_lstm": resnet50_pretrained_imagenet_lstm,
    "resnet50_pretrained_imagenet_lstm_avg_pool": resnet50_pretrained_imagenet_lstm_avg_pool,
    "resnet50_pretrained_imagenet_lstm_custom_loss": resnet50_pretrained_imagenet_lstm_custom_loss,
    "resnet152_pretrained_imagenet_lstm_avg_pool": resnet152_pretrained_imagenet_lstm_avg_pool,
    "resnet152_pretrained_imagenet_lstm_custom_loss": resnet152_pretrained_imagenet_lstm_custom_loss,
    "resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_avg_pool": resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_avg_pool,
    "resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_custom_loss": resnet152_mask_mog2_10channel_pretrained_imagenet_lstm_custom_loss,
    "resnet152_mask_pretrained_imagenet_lstm_avg_pool": resnet152_mask_pretrained_imagenet_lstm_avg_pool,
    "resnet152_mask_pretrained_imagenet_lstm_custom_pool": resnet152_mask_pretrained_imagenet_lstm_custom_pool,
    
    # Mask-based models
    "resnet152_mask_pretrained_imagenet": resnet152_mask_pretrained_imagenet,
    "resnet152_mask_mog2_10channel_pretrained_imagenet": resnet152_mask_mog2_10channel_pretrained_imagenet,

    # OpticalFlow-based models.
    "resnet152_6channel_opticalflow": resnet152_6channel_opticalflow,
    "resnet152_6channel_allpretrained_opticalflow": resnet152_6channel_allpretrained_opticalflow,
    "resnet152_15channel_opticalflow": resnet152_15channel_opticalflow,
    "resnet152_15channel_allpretrained_opticalflow": resnet152_15channel_allpretrained_opticalflow,

    # Hybrid Models with Optical Flow and MOG2 Mask.
    "resnet152_13channel_allpretrained_hybrid": resnet152_13channel_allpretrained_hybrid,
    "resnet152_16channel_allpretrained_hybrid": resnet152_16channel_allpretrained_hybrid,
    "resnet152_7channel_allpretrained_maskopticalflow": resnet152_7channel_allpretrained_maskopticalflow
}


class ModelFactory(object):
    def __init__(self):
        print("Available model architectures are: %s" % AVAILABLE_MODEL_ARCHS.keys())

    def get_model(self, model_arch, input_shape, is_training=False, num_classes=1, learning_rate=0.001):
        return AVAILABLE_MODEL_ARCHS[model_arch](input_shape,
                                                 is_training=is_training,
                                                 num_classes=num_classes,
                                                 learning_rate=learning_rate)
