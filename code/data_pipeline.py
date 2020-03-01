#!python
# -*- coding: utf-8 -*-

"""
Contains the code to build the Data Input Pipeline for TensorFlow models.

"""
import numpy as np
import pandas as pd
import tensorflow as tf


class PipelineGenerator(object):
    """
    Creates a pipeline with the required configuration and image 
    transformations. Takes as input a CSV file which lists the images that needs
    to be loaded and transformed.
    
    NOTE: The dataset must have paths of the images in the sequences stored in 
          columns with names starting from 'image1', 'image2', and so on.
    
    Parameters:
    -----------
    dataset_file: Path to the CSV file containing the list of images and labels.
    
    images_dir: Path to the directory containing the images to be loaded.
    
    sequence_image_count: The number of images in each sequence. Default is 3.
    
    label_name: Name of the column in the CSV file corresponding to the label.
                Default is 'has_animal'.
                
    mode: The string representing the mode in which the data needs to be loaded. 
          For possible modes and definitions, check the 'Attributes' section. 
          Default is MODE_ALL.
          
    image_size: Specifies the size of the input images. Must be provided as a 
                tuple of integers specifying width and height. Default is 
                (224, 224).
          
    image_idx: Used when the selected mode is MODE_SINGLE. Specifies the index 
               of the image that needs to be picked. Must be > 0 and 
               <= sequence_image_count. Default is 1.

    resize: Specifies the size to which the images must be resized. Default is
            None. Must be provided as a tuple of integers specifying width and
            height. If None, no resizing is done.

    perform_shuffle: Specify if the dataset needs to be shuffled. Default: True.

    shuffle_buffer_size: Specifies the buffer size to use to shuffle the CSV
                         records. Check tensorflow.data.Dataset.shuffle() 
                         documentation for more details. Default is 10000.

    kwargs: Any additional keywords argument that needs to be passed to the 
            make_csv_dataset function of TensorFlow.
            
    Attributes:
    -----------
    MODE_ALL: Configuration to make the pipeline return all the images in a 
              dictionary with key as the original column name.
    
    MODE_FLAT_ALL: Configuration to make the pipeline returns all the images of 
                   the sequence one by one.
    
    MODE_SINGLE: Configuration to make the pipeline return only the selected 
                 image from the sequence. Choice of image is specified by the 
                 parameter `image_idx`.
                 
    MODE_SEQUENCE: Configuration to make the pipeline return the sequence of 
                   images as a tensor (array) with a single label.
                   
    MODE_MASK_MOG2_SINGLE: Configuration similar to MODE_SINGLE except the image
                           has an additional mask channel.

    MODE_MASK_MOG2_SEQUENCE: Configuration similar to MODE_SEQUENCE except each 
                             image has an additional mask channel.
                             
    MODE_MASK_MOG2_MULTICHANNEL: Configuration to make the pipeline return all 
                                 the images along with the mask concatenated 
                                 along the depth channel.
            
    """
    # Mode declarations.
    MODE_ALL = "mode_all"

    # Modes for single-image models.
    MODE_FLAT_ALL = "mode_flat_all"
    MODE_SINGLE = "mode_single"

    SINGLE_IMAGE_MODES = [MODE_SINGLE, MODE_FLAT_ALL]

    # Modes for models which use sequence information in some way.
    MODE_SEQUENCE = "mode_sequence"
    MODE_MASK_MOG2_SINGLE = "mode_mask_mog2_single"
    MODE_MASK_MOG2_SEQUENCE = "mode_mask_mog2_sequence"
    MODE_MASK_MOG2_MULTICHANNEL = "mode_mask_mog2_multichannel"
    MODE_OPTICALFLOW_SINGLE = "mode_opticalflow_single"
    MODE_OPTICALFLOW_MULTICHANNEL = "mode_opticalflow_multichannel"  # MODE_OPTICALFLOW_SEQUENCE maybe added in future.
                                                                     # But a different vesion is being run with MODE_SEQUENCE itself.
    MODE_HYBRID_13CHANNEL = "mode_hybrid_13channel"
    MODE_HYBRID_16CHANNEL = "mode_hybrid_16channel"

    MODE_OPTICALFLOWONLY_6CHANNEL = "mode_opticalflowonly_6channel"
    MODE_MASKOPTICALFLOWONLY_7CHANNEL = "mode_maskopticalflowonly_7channel"

    SEQUENCE_MODES = [MODE_SEQUENCE, MODE_MASK_MOG2_SEQUENCE, MODE_MASK_MOG2_SINGLE, MODE_MASK_MOG2_MULTICHANNEL,
                      MODE_OPTICALFLOW_SINGLE, MODE_OPTICALFLOW_MULTICHANNEL,
                      MODE_HYBRID_13CHANNEL, MODE_HYBRID_16CHANNEL,
                      MODE_OPTICALFLOWONLY_6CHANNEL, MODE_MASKOPTICALFLOWONLY_7CHANNEL]

    # Modes which utilize time-step, i.e., which take additional time-step dimensions like (None, 3, 224, 224, 3).
    TIMESTEP_MODES = [MODE_SEQUENCE, MODE_MASK_MOG2_SEQUENCE]

    VALID_MODES = [MODE_ALL, MODE_FLAT_ALL, MODE_SINGLE,
                   MODE_SEQUENCE,
                   MODE_MASK_MOG2_SINGLE, MODE_MASK_MOG2_SEQUENCE, MODE_MASK_MOG2_MULTICHANNEL,
                   MODE_OPTICALFLOW_SINGLE, MODE_OPTICALFLOW_MULTICHANNEL,
                   MODE_HYBRID_13CHANNEL, MODE_HYBRID_16CHANNEL,
                   MODE_OPTICALFLOWONLY_6CHANNEL, MODE_MASKOPTICALFLOWONLY_7CHANNEL]

    def __init__(self, dataset_file, images_dir, sequence_image_count=3,
                 label_name='has_animal', mode=MODE_ALL, image_size=(224, 224),
                 image_idx=1, resize=None, is_training=True,
                 shuffle_buffer_size=10000, **kwargs):
        self._dataset_file = dataset_file
        self._images_dir = images_dir
        self._sequence_image_count = sequence_image_count
        self._label_name = label_name
        self._mode = mode
        self._image_size = image_size
        self._image_idx = image_idx
        self._resize = resize
        self._is_training = is_training
        self._shuffle_buffer_size = shuffle_buffer_size
        self._kwargs = kwargs
        self._AUTOTUNE = tf.data.experimental.AUTOTUNE
        self._size = None

        if self._mode not in self.VALID_MODES:
            raise ValueError("Invalid mode. Please select one from {}."\
                             .format(self.VALID_MODES))
        
        if (self._mode in [self.MODE_SINGLE, self.MODE_MASK_MOG2_SINGLE, self.MODE_OPTICALFLOW_SINGLE] and
            (self._image_idx <= 0 or 
             self._image_idx > self._sequence_image_count)):
            raise IndexError("Image index is out of bounds.")
        
        if self._resize:
            self._image_size = self._resize
            
        self._parser_map = {
            self.MODE_ALL: self._parse_data_all,
            self.MODE_FLAT_ALL: self._parse_data_flat,
            self.MODE_SEQUENCE: self._parse_data_sequence,
            self.MODE_MASK_MOG2_SINGLE: self._parse_data_mask_mog2_single,
            self.MODE_MASK_MOG2_SEQUENCE: self._parse_data_mask_mog2_sequence,
            self.MODE_MASK_MOG2_MULTICHANNEL: self._parse_data_mask_mog2_multichannel,
            self.MODE_OPTICALFLOW_SINGLE: self._parse_data_opticalflow_single,
            self.MODE_OPTICALFLOW_MULTICHANNEL: self._parse_data_opticalflow_multichannel,
            self.MODE_HYBRID_13CHANNEL: self._parse_data_hybrid_13channel,
            self.MODE_HYBRID_16CHANNEL: self._parse_data_hybrid_16channel,
            self.MODE_OPTICALFLOWONLY_6CHANNEL: self._parse_data_opticalflowonly_6channel,
            self.MODE_MASKOPTICALFLOWONLY_7CHANNEL: self._parse_data_maskopticalflowonly_7channel,
            self.MODE_SINGLE: self._parse_data_single
        }

        self._parse_data = self._parser_map[self._mode]



    def _augment_img(self, img, seed, should_skip_color_aug=False):
        
        def flip(x):
            """Flip augmentation

            Args:
                x: Image to flip

            Returns:
                x: Augmented image
            """
            x = tf.image.random_flip_left_right(x, seed=seed)
            return x

        def color(x):
            """Color augmentation

            Args:
                x: Image

            Returns:
                x: Augmented image
            """
            x = tf.image.random_hue(x, 0.08, seed=seed)
            x = tf.image.random_saturation(x, 0.6, 1.6, seed=seed)
            x = tf.image.random_brightness(x, 0.05, seed=seed)
            x = tf.image.random_contrast(x, 0.7, 1.3, seed=seed)
            return x
        
        def zoom(x):
            """Zoom augmentation

            Args:
                x: Image

            Returns:
                x: Augmented image
            """
            # Generate 10 crop settings, ranging from a 1% to 10% crop.
            scales = list(np.arange(0.9, 1.0, 0.02))
            scale = scales[seed % len(scales)]
            
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes = [x1, y1, x2, y2]
            
            # Create different crops for an image
            x = tf.image.crop_and_resize([img], boxes=[boxes], 
                                         box_indices=[0], 
                                         crop_size=self._image_size)
            
            # Squeeze out the final dimension
            x = tf.squeeze(x)
            
            return x

        # Augment images only in training mode.
        if not self._is_training:
            return img

        img = flip(img)

        if seed < 500 and not should_skip_color_aug:
            img = color(img)
        
        if seed >= 250 and seed < 750:
            img = zoom(img)
        
        return tf.clip_by_value(img, 0, 1)
    
    
    def _decode_img(self, img, is_mask=False):
        # XXX: is_mask and num_channels can be merged. Keeping it separate for minimizing code change.
        # Convert the compressed string to a uint8 tensor
        if is_mask:
            img = tf.image.decode_image(img, channels=1)
            img.set_shape(self._image_size + (1,))
        else:    
            img = tf.image.decode_image(img, channels=3)
            img.set_shape(self._image_size + (3,))

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Resize the image to the desired size if needed.
        if self._resize:
            img = tf.image.resize(img, list(self._resize), name="resize-input")

        return img

    
    def _parse_data_all(self, metadata, label):
        data_point = {}
        seed = np.random.randint(1000)
        
        # Read each image and add to dictionary
        for img_num in range(1, self._sequence_image_count + 1):
            img_name = "image" + str(img_num)
            img = tf.io.read_file(tf.strings.join([
                self._images_dir, metadata[img_name]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            data_point[img_name] = img

        return data_point, label
    
    
    def _parse_data_single(self, metadata, label):
        img = tf.io.read_file(tf.strings.join([
                self._images_dir, metadata["image" + str(self._image_idx)]]))
        img = self._decode_img(img)
        
        seed = np.random.randint(1000)
        img = self._augment_img(img, seed)
        return img, label
    
    
    def _parse_data_mask_mog2_single(self, metadata, label):
        # Read the image
        img = tf.io.read_file(tf.strings.join([
                self._images_dir, metadata["image" + str(self._image_idx)]]))
        img = self._decode_img(img)
        
        # Read the mask
        mask = tf.io.read_file(tf.strings.join([self._images_dir, 
                                                metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        
        # Augment the image and the mask
        seed = np.random.randint(1000)
        img = self._augment_img(img, seed)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)
        
        # Append the mask to the image
        final_image = tf.concat([img, mask], axis=2)
        
        return final_image, label

    def _parse_data_flat(self, metadata, label):
        images, labels = [], []
        seed = np.random.randint(1000)
        
        # Read each image and add to list
        for img_num in range(1, self._sequence_image_count + 1):
            img = tf.io.read_file(tf.strings.join([
                self._images_dir, metadata["image" + str(img_num)]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            images.append(img)
            labels.append(label)
        
        return tf.data.Dataset.from_tensor_slices((images, labels))
    
    
    def _parse_data_sequence(self, metadata, label):
        images = []
        seed = np.random.randint(1000)
        
        # Read each image and add to list
        for img_num in range(1, self._sequence_image_count + 1):
            img = tf.io.read_file(tf.strings.join([
                self._images_dir, metadata["image" + str(img_num)]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            images.append(img)
        
        return tf.convert_to_tensor(images), label
    
    
    def _parse_data_mask_mog2_sequence(self, metadata, label):
        images = []
        seed = np.random.randint(1000)
        
        # Read and augment the mask
        mask = tf.io.read_file(tf.strings.join([self._images_dir, 
                                                metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)
        
        # Read each image, augment it, append the mask and add it to the list
        for img_num in range(1, self._sequence_image_count + 1):
            img = tf.io.read_file(tf.strings.join([
                    self._images_dir, metadata["image" + str(img_num)]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            
            # Append the mask to the image
            final_image = tf.concat([img, mask], axis=2)
            images.append(final_image)
            
        return tf.convert_to_tensor(images), label


    def _parse_data_mask_mog2_multichannel(self, metadata, label):
        images = []
        seed = np.random.randint(1000)
        
        # Read each image, augment it and add it to the list
        for img_num in range(1, self._sequence_image_count + 1):
            img = tf.io.read_file(tf.strings.join([
                    self._images_dir, metadata["image" + str(img_num)]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            images.append(img)
            
        # Read and augment the mask. Add it to the list
        mask = tf.io.read_file(tf.strings.join([self._images_dir, 
                                                metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)
        images.append(mask)
        
        # Append all the images and the mask
        final_image = tf.concat(images, axis=2)
        return final_image, label

    def _parse_data_opticalflow_single(self, metadata, label):
        # Read the image
        img = tf.io.read_file(tf.strings.join([self._images_dir, metadata["image" + str(self._image_idx)]]))
        img = self._decode_img(img)

        # Read the opticalflow "average" image.
        opticalflow_avg = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_average']]))
        opticalflow_avg = self._decode_img(opticalflow_avg)  # Treat opticalflow as a normal image since it is RGB image.

        # Augment the image and the mask
        seed = np.random.randint(1000)
        img = self._augment_img(img, seed)
        opticalflow_avg = self._augment_img(opticalflow_avg, seed, should_skip_color_aug=True)

        # Append the opticalflow channels to the image
        final_image = tf.concat([img, opticalflow_avg], axis=2)
        return final_image, label

    def _parse_data_opticalflow_multichannel(self, metadata, label):
        # Note: Provide 'sequence_image_count' as the number of original images, and do not include
        # opticalflow images in it, but provide the right number of channels.
        # List of tuples: [(image_column_name, flag)]
        # The flag determines if the image should_skip_color_aug (mask/optical flow type of image).
        image_names_should_skip_color_aug_flags = []
        for i in range(1, self._sequence_image_count):  # Skip the last image.
            image_names_should_skip_color_aug_flags.append(("image" + str(i), False))
            image_names_should_skip_color_aug_flags.append(("opticalflowGF_" + str(i), True))  # Need to skip color aug, as this is opticalflow.
        image_names_should_skip_color_aug_flags.append(("image" + str(self._sequence_image_count), False))  # Include last image here.

        # Read each image, augment it if it not opticalflow image and add it to the list.
        images = []
        seed = np.random.randint(1000)
        for column_name, should_skip_color_aug in image_names_should_skip_color_aug_flags:
            img = tf.io.read_file(tf.strings.join([self._images_dir, metadata[column_name]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed, should_skip_color_aug=should_skip_color_aug)
            images.append(img)

        # Append all the images and the mask
        final_image = tf.concat(images, axis=2)
        return final_image, label

    def _parse_data_hybrid_13channel(self, metadata, label):
        images = []
        seed = np.random.randint(1000)

        # Read each image, augment it and add it to the list
        for img_num in range(1, self._sequence_image_count + 1):
            img = tf.io.read_file(tf.strings.join([self._images_dir, metadata["image" + str(img_num)]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed)
            images.append(img)

        # Read and augment the OpticalFlow-Average. Add it to the list
        opticalflow_avg = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_average']]))
        opticalflow_avg = self._decode_img(opticalflow_avg)
        opticalflow_avg = self._augment_img(opticalflow_avg, seed, should_skip_color_aug=True)
        images.append(opticalflow_avg)

        # Read and augment the mask. Add it to the list
        mask = tf.io.read_file(tf.strings.join([self._images_dir, metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)
        images.append(mask)

        # Append all the images and the mask
        final_image = tf.concat(images, axis=2)
        return final_image, label

    def _parse_data_hybrid_16channel(self, metadata, label):
        # Note: Provide 'sequence_image_count' as the number of original images, and do not include
        # opticalflow images in it, but provide the right number of channels.
        # List of tuples: [(image_column_name, flag)]
        # The flag determines if the image should_skip_color_aug (mask/optical flow type of image).
        image_names_should_skip_color_aug_flags = []
        for i in range(1, self._sequence_image_count):  # Skip the last image.
            image_names_should_skip_color_aug_flags.append(("image" + str(i), False))
            image_names_should_skip_color_aug_flags.append(("opticalflowGF_" + str(i), True))  # Need to skip color aug, as this is opticalflow.
        image_names_should_skip_color_aug_flags.append(("image" + str(self._sequence_image_count), False))  # Include last image here.

        # Read each image, augment it if it not opticalflow image and add it to the list.
        images = []
        seed = np.random.randint(1000)
        for column_name, should_skip_color_aug in image_names_should_skip_color_aug_flags:
            img = tf.io.read_file(tf.strings.join([self._images_dir, metadata[column_name]]))
            img = self._decode_img(img)
            img = self._augment_img(img, seed, should_skip_color_aug=should_skip_color_aug)
            images.append(img)

        # Read and augment the mask. Add it to the list
        mask = tf.io.read_file(tf.strings.join([self._images_dir, metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)
        images.append(mask)

        # Append all the images and the mask
        final_image = tf.concat(images, axis=2)
        return final_image, label

    def _parse_data_opticalflowonly_6channel(self, metadata, label):
        seed = np.random.randint(1000)
        # Read the opticalflow 1 and 2 images.
        opticalflow_1 = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_1']]))
        opticalflow_1 = self._decode_img(opticalflow_1)  # Treat opticalflow as a normal image since it is RGB image.
        opticalflow_1 = self._augment_img(opticalflow_1, seed, should_skip_color_aug=True)

        opticalflow_2 = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_2']]))
        opticalflow_2 = self._decode_img(opticalflow_2)  # Treat opticalflow as a normal image since it is RGB image.
        opticalflow_2 = self._augment_img(opticalflow_2, seed, should_skip_color_aug=True)

        # Append the opticalflow channels to the image
        final_image = tf.concat([opticalflow_1, opticalflow_2], axis=2)
        return final_image, label

    def _parse_data_maskopticalflowonly_7channel(self, metadata, label):
        seed = np.random.randint(1000)
        # Read the opticalflow 1 and 2 images.
        opticalflow_1 = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_1']]))
        opticalflow_1 = self._decode_img(opticalflow_1)  # Treat opticalflow as a normal image since it is RGB image.
        opticalflow_1 = self._augment_img(opticalflow_1, seed, should_skip_color_aug=True)

        opticalflow_2 = tf.io.read_file(tf.strings.join([self._images_dir, metadata['opticalflowGF_2']]))
        opticalflow_2 = self._decode_img(opticalflow_2)  # Treat opticalflow as a normal image since it is RGB image.
        opticalflow_2 = self._augment_img(opticalflow_2, seed, should_skip_color_aug=True)

        # Read and augment the mask. Add it to the list
        mask = tf.io.read_file(tf.strings.join([self._images_dir, metadata['mask_MOG2']]))
        mask = self._decode_img(mask, is_mask=True)
        mask = self._augment_img(mask, seed, should_skip_color_aug=True)

        # Append the opticalflow channels to the image
        final_image = tf.concat([opticalflow_1, opticalflow_2, mask], axis=2)
        return final_image, label

    def get_size(self):
        if self._size is None:
            print("Size cannot be determined before the 'get_pipeline' function call. Returning None.")
        return self._size

    def get_pipeline(self):
        """
        Returns a pipeline that was constructed using the parameters specified.
        
        Returns:
        --------
        dataset_images: A tensorflow.data.Dataset pipeline object.
        
        """
        # Create a dataset with records from the CSV file.
        data_csv = pd.read_csv(self._dataset_file)

        # Set the size attribute.
        self._size = len(data_csv)
        if self._mode == self.MODE_FLAT_ALL:
            self._size = self._size * self._sequence_image_count

        image_col_names = ["image" + str(img_num) for img_num in range(1, self._sequence_image_count + 1)]
        mask_columns = [mask_col for mask_col in data_csv.columns if (mask_col.startswith('mask') or mask_col.startswith('opticalflow'))]
        image_col_names.extend(mask_columns)
        labels = data_csv[[self._label_name]]
        file_paths = data_csv[image_col_names]
        dataset_files = tf.data.Dataset.from_tensor_slices((file_paths.to_dict('list'), labels.values.reshape(-1, )))

        # Parse the data and load the images.
        if self._mode == self.MODE_FLAT_ALL:
            dataset_images = dataset_files.flat_map(self._parse_data)
        else:
            dataset_images = dataset_files.map(self._parse_data, num_parallel_calls=self._AUTOTUNE)

        if self._is_training:
            dataset_images = dataset_images.shuffle(buffer_size=self._shuffle_buffer_size, reshuffle_each_iteration=True)
            dataset_images = dataset_images.repeat()
            print("Note: The dataset is being prepared for training mode. "
                  "It has been shuffled, and repeated indefinitely.")

        return dataset_images
