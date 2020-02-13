#!python
# -*- coding: utf-8 -*-

"""
This module contains the code to train the baseline (single-image based) model.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from data_pipeline import PipelineGenerator
from models import ModelFactory
from tensorflow import keras

import argparse
import os
import time


def get_args():
    parser = argparse.ArgumentParser(description="Run the the baseline model training.")

    parser.add_argument('--train-meta-file',
                        required=True,
                        help="The path to the file containing the training images metadata.")
    parser.add_argument('--val-meta-file',
                        required=True,
                        help="The path to the file containing the validation images metadata.")
    parser.add_argument('--images-dir',
                        required=True,
                        help="The path to the directory containing the images.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the output dir to which the trained model files need to be written.")
    parser.add_argument('--model-arch',
                        required=True,
                        help="The name of the model architecture which is present in the 'models' module.")
    parser.add_argument('--data-pipeline-mode',
                        required=True,
                        help="The mode to be used for the data pipeline.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help="The number of datapoints in a training batch. This size must fit within the GPU memory. "
                             "Default: 64.")
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="The number of training epochs. Default: 100.")
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help="The constant learning rate to be used for training. Default: 0.001.")

    parser.add_argument('--image-size',
                        type=int,
                        default=224,
                        help="The length of the side for the 'square' images present in the images-dir. Default: 224.")

    args = parser.parse_args()

    return args


def train(train_metadata_file_path,
          val_metadata_file_path,
          images_dir_path,
          out_dir,
          model_arch,
          num_classes,
          label_name=None,
          sequence_image_count=1,
          data_pipeline_mode="mode_flat_all",
          class_weight=None,
          epochs=100,
          batch_size=32,
          learning_rate=0.001,
          input_size=(224, 224, 3)):
    """
    Train a VGG16 model based on single image.

    :param train_metadata_file_path: The path to the metadata '.csv' file containing training image names.
    :param val_metadata_file_path: The path to the metadata '.csv' file containing validation image names.
    :param images_dir_path: The path containing the images.
    :param out_dir: The path to which the saved models need to be written.
    :param model_arch: The model architecture provided as string, which are present in the 'models' module.
    :param num_classes: The number of classes present in the data. If num_classes=1, it requires the 'label_name'.
    :param label_name: Required if num_classes=1. The name of the label to pick from the data.
    :param sequence_image_count: The number of images in the sequence dataset. Default: 1.
    :param data_pipeline_mode: The mode of the data pipeline. Default: "mode_flat_all".
    :param class_weight: The class_weights for imbalanced data. Example: {0: 1.0, 1: 0.5}, if class "0" is twice less
        represented than class "1" in your data. Default: None.
    :param epochs: The maximum number of epochs to be trained. Note that the model maybe early-stopped. Default: 100.
    :param batch_size: The batch size used for the data. Ensure that it fits within the GPU memory. Default: 32.
    :param learning_rate: The constant learning rate to be used for the Adam optimizer. Default: 0.001.
    :param input_size: The shape of the tensors returned by the data pipeline mode. Default: (224, 224, 3).

    """
    if num_classes == 1 and label_name is None:
        raise ValueError("Since num_classes equals 1, the label_name must be provided.")

    train_data_epoch_subdivisions = 4
    early_stop_monitor = "val_loss"
    early_stop_min_delta = 0.005
    early_stop_patience = 2 * train_data_epoch_subdivisions  # One run through the train dataset.
    prefetch_buffer_size = 3  # Can be also be set to tf.data.experimental.AUTOTUNE

    os.makedirs(out_dir)

    # Build model architecture.
    model_factory = ModelFactory()
    model = model_factory.get_model(model_arch,
                                    input_size,
                                    is_training=True,
                                    num_classes=num_classes,
                                    learning_rate=learning_rate)
    print("Created the model architecture: %s" % model.name)
    model.summary()

    # Prepare the training dataset.
    print("Preparing training and validation datasets.")
    train_data_pipeline = PipelineGenerator(train_metadata_file_path,
                                            images_dir_path,  # XXX: This function calls requires this path to end with slash.
                                                              # This needs to be handled in the PipelineGenerator.
                                            is_training=True,
                                            sequence_image_count=sequence_image_count,
                                            label_name=label_name,
                                            mode=data_pipeline_mode)
    train_dataset = train_data_pipeline.get_pipeline()
    train_dataset = train_dataset.batch(batch_size).prefetch(prefetch_buffer_size)

    # Prepare the validation dataset
    val_data_pipeline = PipelineGenerator(val_metadata_file_path,
                                          images_dir_path,
                                          is_training=False,
                                          sequence_image_count=sequence_image_count,
                                          label_name=label_name,
                                          mode=data_pipeline_mode)
    val_dataset = val_data_pipeline.get_pipeline()
    val_dataset = val_dataset.batch(batch_size).prefetch(prefetch_buffer_size)

    # TODO: Find a way to log the activation maps, either during training, or after the training has completed.

    # Prepare the callbacks.
    print("Preparing Tensorflow Keras Callbacks.")
    earlystop_callback = keras.callbacks.EarlyStopping(monitor=early_stop_monitor,
                                                       min_delta=early_stop_min_delta,
                                                       patience=early_stop_patience)

    best_model_checkpoint_acc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best_model_dir-acc.ckpt"),
                                                                         mode='max',
                                                                         monitor='val_accuracy',
                                                                         save_best_only=True,
                                                                         save_weights_only=False,
                                                                         verbose=1)
    best_model_checkpoint_loss_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best_model_dir-loss.ckpt"),
                                                                          mode='min',
                                                                          monitor='val_loss',
                                                                          save_best_only=True,
                                                                          save_weights_only=False,
                                                                          verbose=1)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir, "TBGraph"),
                                                       write_graph=True,
                                                       write_images=True)

    callbacks = [earlystop_callback,
                 best_model_checkpoint_acc_callback,
                 best_model_checkpoint_loss_callback,
                 tensorboard_callback]

    # Start model training.
    # Defining an 'epoch' to be a quarter of the train dataset.
    num_train_samples = train_data_pipeline.get_size()
    num_val_samples = val_data_pipeline.get_size()
    # Number of batches per one run through the train dataset.
    num_training_steps_per_whole_dataset = int(num_train_samples / batch_size)
    num_val_steps_per_whole_dataset = int(num_val_samples / batch_size)
    steps_per_epoch = int(num_training_steps_per_whole_dataset / train_data_epoch_subdivisions)
    max_num_sub_epochs = epochs * train_data_epoch_subdivisions
    max_train_steps = int(max_num_sub_epochs * steps_per_epoch)

    print("Number of train samples: %s, which correspond to  ~%s batches for one complete run through the "
          "train dataset. Number of validation samples: %s, which correspond to ~%s batches for complete iteration. "
          "Considering a 1/%s fraction of the train dataset as an epoch (steps_per_epoch: %s, max_num_sub_epochs: %s) "
          "after which validation and model checkpoints are saved. Running training for a maximum of %s steps. "
          "Early stopping has been set based on '%s' of min_delta of %s with a patience of %s."
          % (num_train_samples, num_training_steps_per_whole_dataset,
             num_val_samples, num_val_steps_per_whole_dataset,
             train_data_epoch_subdivisions, steps_per_epoch, max_num_sub_epochs
             max_train_steps,
             early_stop_monitor, early_stop_min_delta, early_stop_patience))

    print("\nStarting the model training.")
    start_time = time.time()

    model.fit(train_dataset,
              epochs=max_num_sub_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_dataset,
              validation_steps=num_val_steps_per_whole_dataset,
              callbacks=callbacks,
              class_weight=class_weight)

    time_taken = time.time() - start_time
    print("Training completed and the output has been saved in %s. Time taken: %s seconds." % (out_dir, time_taken))


if __name__ == "__main__":
    args = get_args()

    num_classes = 1
    label_name = "has_animal"
    # TODO: Calculate this automatically probably in the data pipeline, and set it as a property.
    class_weight = {0: 1, 1: 0.21}  # The number images found in train metadata file - {0: 10738, 1: 51426}
    sequence_image_count = 3

    if args.data_pipeline_mode == "MODE_SEQUENCE":
        input_size = (sequence_image_count, args.image_size, args.image_size, 3)
    else:
        input_size = (args.image_size, args.image_size, 3)

    train(args.train_meta_file,
          args.val_meta_file,
          args.images_dir,
          args.out_dir,
          args.model_arch,
          num_classes,
          label_name=label_name,
          sequence_image_count=sequence_image_count,
          data_pipeline_mode=args.data_pipeline_mode,
          class_weight=class_weight,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          input_size=input_size)
