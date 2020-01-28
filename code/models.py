
"""
This module contains the Tensorflow Keras models.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from tensorflow.keras import layers


def _add_conv_block(num_filters, inputs, is_training, name_prefix, kernel_regularizer=None):
    """Adds a Batchnorm enabled convolution block.
    """
    conv = layers.Conv2D(num_filters,
                         (3, 3),
                         activation=None,
                         padding="same",
                         kernel_regularizer=kernel_regularizer,
                         name="%s-Conv" % name_prefix)(inputs)
    batchnorm = layers.BatchNormalization(name="%s-BN" % name_prefix)(conv, training=is_training)
    relu = layers.ReLU(name="%s-Relu" % name_prefix)(batchnorm)
    return relu


def _add_dense_block(units, inputs, is_training, name_prefix, kernel_regularizer=None, dropout_rate=0.5):
    """Adds a Batchnorm and Dropout enabled dense block.
    """
    dense = layers.Dense(units, activation=None, name="%s-Dense" % name_prefix)(inputs)
    batchnorm = layers.BatchNormalization(name="%s-BN" % name_prefix)(dense, training=is_training)
    relu = layers.ReLU(name="%s-Relu" % name_prefix)(batchnorm)
    dropout = layers.Dropout(dropout_rate, name="%s-Dropout_%s" % (name_prefix, dropout_rate))(relu, training=is_training)
    return dropout


def vgg16(input_shape, is_training, num_classes=1):
    """
    Builds a vgg16 model with Batchnorm and Dropout layers.

    :param input_shape: THe shape of the input images (do not consider the batch-dimension).
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
        predictions = layers.Dense(1, activation="sigmoid", name="prediction")(fc2)
    else:
        predictions = layers.Dense(num_classes, activation="softmax", name="prediction")(fc2)

    model = keras.Model(inputs=inputs, outputs=predictions, name="vgg16_BN_model")

    return model


def train():
    # TODO: Move it to train pipeline.
    model = vgg16((256, 256, 3), True, num_classes=1)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Prepare the training dataset
    train_dataset = get_training_dataset # TODO
    train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(64)
    class_weight = {0: 1., 1: 0.5}  # if class "0" is twice less represented than class "1" in your data. # TODO

    # Prepare the validation dataset
    val_dataset = get_val_dataset # TODO
    val_dataset = val_dataset.batch(64)

    earlystop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
    best_model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=filepath, # TODO
                                                                     mode='max',
                                                                     monitor='val_acc',
                                                                     save_best_only=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./Graph', # TODO
                                                       write_graph=True,
                                                       write_images=True)
    # TODO: Find a way to log the activation maps, either during training, or after the training has completed.

    model.fit(train_dataset, epochs=3,
              # Only run validation using the first 10 batches of the dataset
              # using the `validation_steps` argument
              validation_data=val_dataset, validation_steps=10,  # TODO: All steps for 1 epoch.
              callbacks=[earlystop_callback, best_model_checkpoint_callback, tensorboard_callback],
              class_weight=class_weight)
