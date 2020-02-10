#!python
# -*- coding: utf-8 -*-

"""
This module contains the code to train the baseline (single-image based) model.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from data_pipeline import PipelineGenerator
from models import vgg16
from tensorflow import keras

import os
import tensorflow as tf
import time


def train(train_metadata_file_path,
          val_metadata_file_path,
          images_dir_path,
          out_dir,
          class_weight=None,
          epochs=100,
          batch_size=32,
          learning_rate=0.001,
          image_size=(224, 224, 3)):
    """
    Train a VGG16 model based on single image.

    :param train_metadata_file_path: The path to the metadata '.csv' file containing training image names.
    :param val_metadata_file_path: The path to the metadata '.csv' file containing validation image names.
    :param images_dir_path: The path containing the images.
    :param out_dir: The path to which the saved models need to be written.
    :param class_weight: The class_weights for imbalanced data. Example: {0: 1.0, 1: 0.5}, if class "0" is twice less
        represented than class "1" in your data. Default: None.
    :param epochs: The maximum number of epochs to be trained. Note that the model maybe early-stopped. Default: 100.
    :param batch_size: The batch size used for the data. Ensure that it fits within the GPU memory. Default: 32.
    :param learning_rate: The constant learning rate to be used for the Adam optimizer. Default: 0.001.
    :param image_size: The shape of the images present in the images_dir_path. Default: (224, 224, 3).

    """
    os.makedirs(out_dir)

    # Build model architecture.
    print("Building VGG16 model architecture.")
    model = vgg16(image_size, is_training=True, num_classes=1)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("The VGG16 model:\n%s" % model.summary())

    # Prepare the training dataset.
    print("Preparing training and validation datasets.")
    train_dataset = PipelineGenerator(train_metadata_file_path,
                                      images_dir_path,
                                      perform_shuffle=True,
                                      sequence_image_count=3,
                                      label_name="has_animal",
                                      mode="mode_flat_all")
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare the validation dataset
    val_dataset = PipelineGenerator(val_metadata_file_path,
                                    images_dir_path,
                                    perform_shuffle=False,
                                    sequence_image_count=3,
                                    label_name="has_animal",
                                    mode="mode_flat_all")
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # TODO: Find a way to log the activation maps, either during training, or after the training has completed.

    print("Finding the number of training batches.")
    start_time = time.time()
    num_training_steps = 0
    for _ in train_dataset:
        num_training_steps += 1
    print("Total number of training batches: %s. Time taken for training dataset iteration: %s seconds."
          % (num_training_steps, time.time() - start_time))

    print("Finding the number of validation batches.")
    start_time = time.time()
    num_validation_steps = 0
    for _ in val_dataset:
        num_validation_steps += 1
    print("Total number of validation batches: %s. Time taken for validation dataset iteration: %s seconds."
          % (num_validation_steps, time.time() - start_time))

    # Prepare the callbacks.
    print("Preparing Tensorflow Keras Callbacks.")
    earlystop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
    best_model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best_model_dir"),
                                                                     mode='max',
                                                                     monitor='val_acc',
                                                                     save_best_only=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir, "TBGraph"),
                                                       write_graph=True,
                                                       write_images=True)

    # Start model training.
    print("Starting the model training.")
    start_time = time.time()
    model.fit(train_dataset, epochs=epochs,
              # Only run validation using the first 10 batches of the dataset using the `validation_steps` argument.
              validation_data=val_dataset, validation_steps=num_validation_steps,  # All steps for 1 epoch.
              callbacks=[earlystop_callback, best_model_checkpoint_callback, tensorboard_callback],
              class_weight=class_weight)

    time_taken = time.time() - start_time
    print("Training completed and the output has been saved in %s. Time taken: %s seconds." % (out_dir, time_taken))


if __name__ == "__main__":
    model_name = "baseline_1"
    train_metadata_file = "../data/final_dataset_train.csv"
    val_metadata_file = "../data/final_dataset_val.csv"
    wellington_images_dir = "../../dataset/images-resized/"  # Ensure the slash at the end.
    image_size = (224, 224, 3)
    out_dir = "../models/%s" % model_name
    class_weight = {0: 1, 1: 0.21}  # The number images found in train metadata file - {0: 10738, 1: 51426}
    batch_size = 32
    epochs = 100
    learning_rate = 0.001

    train(train_metadata_file,
          val_metadata_file,
          wellington_images_dir,
          out_dir,
          class_weight=class_weight,
          epochs=epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          image_size=image_size)
