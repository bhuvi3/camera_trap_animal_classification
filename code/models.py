# -*- coding: utf-8 -*-

"""
This module contains the Tensorflow Keras models.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


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
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=loss, metrics=['accuracy'])

    return model


def vgg16_pretrained_imagenet(input_shape, is_training=False, num_classes=1, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape, name='input')

    model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
    output_vgg16_conv = model_vgg16_conv(input)

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
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=loss, metrics=['accuracy'])

    return model


# The dictionary mapping model names to model architecture functions.
# Ensure that the name of the model architecture matches with the model's 'name' attribute.
AVAILABLE_MODEL_ARCHS = {
    "vgg16_batchnorm": vgg16_batchnorm,
    "vgg16_pretrained_imagenet": vgg16_pretrained_imagenet,
}


class ModelFactory(object):
    def __init__(self):
        print("Available model architectures are: %s" % AVAILABLE_MODEL_ARCHS.keys())

    def get_model(self, model_arch, input_shape, is_training=False, num_classes=1, learning_rate=0.001):
        return AVAILABLE_MODEL_ARCHS[model_arch](input_shape,
                                                 is_training=is_training,
                                                 num_classes=num_classes,
                                                 learning_rate=learning_rate)