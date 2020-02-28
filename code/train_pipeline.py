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
    parser.add_argument('--patience',
                        type=int,
                        default=2,
                        help="The number of epochs (full train dataset) to wait before early stopping. Default: 2.")
    parser.add_argument('--min-delta-auc',
                        type=float,
                        default=0.01,
                        help="The minimum delta of validation auc for early stopping after patience. Default: 0.01.")
    parser.add_argument('--image-size',
                        type=int,
                        default=224,
                        help="The length of the side for the 'square' images present in the images-dir. Default: 224.")
    parser.add_argument('--sequence-image-count',
                        type=int,
                        default=3,
                        help="The number of images in the sequence which needs to be input across time-steps. Default: 3.")
    parser.add_argument('--num-channels',
                        type=int,
                        default=3,
                        help="The number of input channels expected by the data pipeline. Default: 3.")

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
          whole_epochs=100,
          batch_size=32,
          learning_rate=0.001,
          patience=2,
          min_delta_auc=0.01,
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
    :param whole_epochs: The maximum number of epochs to be trained. Note that the model maybe early-stopped. Default: 100.
    :param batch_size: The batch size used for the data. Ensure that it fits within the GPU memory. Default: 32.
    :param learning_rate: The constant learning rate to be used for the Adam optimizer. Default: 0.001.
    :param patience: The number of epochs (full train dataset) to wait before early stopping. Default: 2.
    :param min_delta_auc: The minimum delta of validation auc for early stopping after patience. Default: 0.01.
    :param input_size: The shape of the tensors returned by the data pipeline mode. Default: (224, 224, 3).

    """
    if num_classes == 1 and label_name is None:
        raise ValueError("Since num_classes equals 1, the label_name must be provided.")

    train_data_epoch_subdivisions = 4
    early_stop_monitor = "val_auc"
    early_stop_min_delta = min_delta_auc
    early_stop_patience = patience * train_data_epoch_subdivisions  # One run through the train dataset.
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


    # XXX: We use the HDF5 method to store the sequence models due to a bug in tensorflow TimeDistributed wrapper
    if data_pipeline_mode in PipelineGenerator.TIMESTEP_MODES:
        model_extension = ".h5"
    else:
        model_extension = ".ckpt"

    best_model_checkpoint_auc_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best_model_dir-auc"+model_extension),
                                                                         mode='max',
                                                                         monitor='val_auc',
                                                                         save_best_only=True,
                                                                         save_weights_only=False,
                                                                         verbose=1)
    best_model_checkpoint_loss_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best_model_dir-loss"+model_extension),
                                                                          mode='min',
                                                                          monitor='val_loss',
                                                                          save_best_only=True,
                                                                          save_weights_only=False,
                                                                          verbose=1)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir, "TBGraph"),
                                                       write_graph=True,
                                                       write_images=True)

    callbacks = [earlystop_callback,
                 best_model_checkpoint_auc_callback,
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
    max_num_epochs = int(whole_epochs * train_data_epoch_subdivisions)
    max_train_steps = int(max_num_epochs * steps_per_epoch)

    print("Number of train samples: %s, which correspond to  ~%s batches for one complete run through the "
          "train dataset. Number of validation samples: %s, which correspond to ~%s batches for complete iteration. "
          "Considering a 1/%s fraction of the train dataset as an epoch (steps_per_epoch: %s) "
          "after which validation and model checkpoints are saved. Running training for a maximum of %s steps, "
          "which correspond to max_num_epochs: %s (whole_epochs: %s). "
          "Early stopping has been set based on '%s' of min_delta of %s with a patience of %s."
          % (num_train_samples, num_training_steps_per_whole_dataset,
             num_val_samples, num_val_steps_per_whole_dataset,
             train_data_epoch_subdivisions, steps_per_epoch,
             max_train_steps,
             max_num_epochs, whole_epochs,
             early_stop_monitor, early_stop_min_delta, early_stop_patience))

    print("\nStarting the model training.")
    start_time = time.time()

    model.fit(train_dataset,
              epochs=max_num_epochs,
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

    # Default num_channels for backward compatibility.
    if not args.num_channels:
        if args.data_pipeline_mode == PipelineGenerator.MODE_MASK_MOG2_SINGLE:
            args.num_channels = 4
        if args.data_pipeline_mode == PipelineGenerator.MODE_MASK_MOG2_MULTICHANNEL:
            args.num_channels = 10

    if args.data_pipeline_mode in PipelineGenerator.TIMESTEP_MODES:
        input_size = (args.sequence_image_count, args.image_size, args.image_size, args.num_channels)
    else:
        input_size = (args.image_size, args.image_size, args.num_channels)

    train(args.train_meta_file,
          args.val_meta_file,
          args.images_dir,
          args.out_dir,
          args.model_arch,
          num_classes,
          label_name=label_name,
          sequence_image_count=args.sequence_image_count,
          data_pipeline_mode=args.data_pipeline_mode,
          class_weight=None,
          whole_epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          patience=args.patience,
          min_delta_auc=args.min_delta_auc,
          input_size=input_size)
