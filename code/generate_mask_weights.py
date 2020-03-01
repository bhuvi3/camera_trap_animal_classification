import os

import numpy as np
import tensorflow as tf


def save_weights_resnet152_10channel():
    # Initialize configuration
    required_input_shape = (7, 7, 10, 64)
    output_file_prefix = "resnet152_10channel"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)
    new_weights[:, :, :3, :] = input_layer_weights
    new_weights[:, :, 3:6, :] = input_layer_weights
    new_weights[:, :, 6:9, :] = input_layer_weights
    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join(os.getcwd(), 'data', output_file_prefix + "_mask_weights.npy"), weights)


def save_weights_resnet152_6channel(allpretrained=False):
    # Initialize configuration
    required_input_shape = (7, 7, 6, 64)
    output_file_prefix = "resnet152_6channel"
    if allpretrained:
        output_file_prefix = output_file_prefix + "_allpretrained"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)
    new_weights[:, :, :3, :] = input_layer_weights
    if allpretrained:
        new_weights[:, :, 3:6, :] = input_layer_weights

    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join("..", 'data', output_file_prefix + "_opticalflow_weights.npy"), weights)


def save_weights_resnet152_15channel(allpretrained=False):
    # Initialize configuration
    required_input_shape = (7, 7, 15, 64)
    output_file_prefix = "resnet152_15channel"
    if allpretrained:
        output_file_prefix = output_file_prefix + "_allpretrained"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)

    new_weights[:, :, :3, :] = input_layer_weights          # First image.

    if allpretrained:
        new_weights[:, :, 3:6, :] = input_layer_weights     # OpticalFlow-1_2.

    new_weights[:, :, 6:9, :] = input_layer_weights         # Second image.

    if allpretrained:
        new_weights[:, :, 9:12, :] = input_layer_weights    # OpticalFlow-2_3.

    new_weights[:, :, 12:15, :] = input_layer_weights       # Third image.

    # Reassign new weights.
    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join("..", 'data', output_file_prefix + "_opticalflow_weights.npy"), weights)


def save_weights_resnet152_13channel(allpretrained=False):
    """
    image_1 (3) + image_2 (3) + image_3 (3) + opticalflow_average (3) + MOG2_mask (1)

    """
    # Initialize configuration
    required_input_shape = (7, 7, 13, 64)
    output_file_prefix = "resnet152_13channel"
    if allpretrained:
        output_file_prefix = output_file_prefix + "_allpretrained"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)

    new_weights[:, :, :3, :] = input_layer_weights          # First image.
    new_weights[:, :, 3:6, :] = input_layer_weights         # Second image.
    new_weights[:, :, 6:9, :] = input_layer_weights         # Third image.

    if allpretrained:
        new_weights[:, :, 9:12, :] = input_layer_weights    # OpticalFlow-average.
                                                            # Mask always uses newly initialized weights.

    # Reassign new weights.
    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join("..", 'data', output_file_prefix + "_hybrid_weights.npy"), weights)


def save_weights_resnet152_16channel(allpretrained=False):
    """
    image_1 (3) + opticalflow_1 (3) + image_2 (3) + opticalflow_2 (3) + image_3 (3) + MOG2_mask (1)

    """
    # Initialize configuration
    required_input_shape = (7, 7, 16, 64)
    output_file_prefix = "resnet152_16channel"
    if allpretrained:
        output_file_prefix = output_file_prefix + "_allpretrained"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)

    new_weights[:, :, :3, :] = input_layer_weights          # First image.

    if allpretrained:
        new_weights[:, :, 3:6, :] = input_layer_weights     # OpticalFlow-1_2.

    new_weights[:, :, 6:9, :] = input_layer_weights         # Second image.

    if allpretrained:
        new_weights[:, :, 9:12, :] = input_layer_weights    # OpticalFlow-2_3.

    new_weights[:, :, 12:15, :] = input_layer_weights       # Third image.
                                                            # Mask always uses newly initialized weights.

    # Reassign new weights.
    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join("..", 'data', output_file_prefix + "_hybrid_weights.npy"), weights)


def save_weights_resnet152_7channel(allpretrained=False):
    # Initialize configuration
    required_input_shape = (7, 7, 7, 64)
    output_file_prefix = "resnet152_7channel"
    if allpretrained:
        output_file_prefix = output_file_prefix + "_allpretrained"

    # Initialize a model of choice
    model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

    # Get the weights of the model
    weights = model_pretrained_conv.get_weights()
    input_layer_weights = weights[0]

    print("Changing weights of the input layer from", input_layer_weights.shape, "to", required_input_shape)

    # Change the weights to desired shape
    new_weights = np.random.normal(0, 0.001, required_input_shape)
    new_weights[:, :, :3, :] = input_layer_weights
    if allpretrained:
        new_weights[:, :, 3:6, :] = input_layer_weights

    # 7th Channel for Mask uses the randomly initialized weights. Therefore, leave it as it is.

    weights[0] = new_weights

    # Save the new weights
    np.save(os.path.join("..", 'data', output_file_prefix + "_maskopticalflow_weights.npy"), weights)
