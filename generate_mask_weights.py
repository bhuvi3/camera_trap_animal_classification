import os

import numpy as np
import tensorflow as tf


# Initialize configuration
required_input_shape = (7, 7, 4, 64)
output_file_prefix = "resnet152"

# Initialize a model of choice
model_pretrained_conv = tf.keras.applications.ResNet152(weights='imagenet', 
                                                        include_top=False)

# Get the weights of the model
weights = model_pretrained_conv.get_weights()
input_layer_weights = weights[0]

print("Changing weights of the input layer from", input_layer_weights.shape, 
      "to", required_input_shape)

# Change the weights to desired shape
new_weights = np.random.normal(0, 0.001, required_input_shape)
new_weights[:, :, :3, :] = input_layer_weights
weights[0] = new_weights

# Save the new weights
np.save(os.path.join(os.getcwd(), 'data', 
                     output_file_prefix + "_mask_weights.npy"), weights)
