import tensorflow as tf


class PipelineGenerator(object):
    """
    Creates a pipeline with the required configuration and image transformations
    (yet to come). Takes as input a CSV file which lists the images that needs
    to be loaded and transformed.
    
    Parameters:
    -----------
    dataset_file: Path to the CSV file containing the list of images and labels.
    
    images_dir: Path to the directory containing the images to be loaded.
    
    label_name: Name of the column in the CSV file corresponding to the label.
                Default is 'has_animal'.
                
    resize: Specifies the size to which the images must be resized. Default is
            None. Must be provided as a list of integers specifying width and
            height.

    mode: The string representing the mode in which the data needs to be loaded. Default: "all".
        Supported modes:
            - "all": A datapoint contains all three images ('image1', 'image2', 'image3') in the sequence
                as a dictionary.
            - "single-all": A datapoint contains a single image, but all images in the sequence are considered.
            - "single-image1": A datapoint contains a single image, the first image in the sequence.
            - "single-image2": A datapoint contains a single image, the second image in the sequence.
            - "single-image3": A datapoint contains a single image, the third image in the sequence.

    kwargs: Any additional keywords argument that needs to be passed to the 
            make_csv_dataset function of TensorFlow.
    
    """
    
    def __init__(self, dataset_file, images_dir, label_name='has_animal', mode="all", resize=None, **kwargs):
        self._dataset_file = dataset_file
        self._images_dir = images_dir
        self._label_name = label_name
        self._mode = mode
        self._resize = resize
        self._kwargs = kwargs
        self._AUTOTUNE = tf.data.experimental.AUTOTUNE


    def _decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Resize the image to the desired size if needed.
        if self._resize:
            img = tf.image.resize(img, self._resize, name="resize-input")

        return img

    
    def _parse_data(self, metadata, label):
        if self._mode == "all":
            data_point = {}

            # Read each image and add to dictionary
            for img_name in ['image1', 'image2', 'image3']:
                img = tf.io.read_file(tf.strings.join([self._images_dir, metadata[img_name]])[0])
                img = self._decode_img(img)
                data_point[img_name] = img

            return data_point, label

        elif self._mode == "single-all":
            # Note that this mode must be used in a flat-map.
            datapoints = []
            for img_name in ['image1', 'image2', 'image3']:
                img = tf.io.read_file(tf.strings.join([self._images_dir, metadata[img_name]])[0])
                img = self._decode_img(img)
                datapoints.append((img, label))

            return datapoints

        elif self._mode.startswith("single-"):
            image_name = self._mode.split('-')[-1]
            img = tf.io.read_file(tf.strings.join([self._images_dir, metadata[image_name]])[0])
            img = self._decode_img(img)
            return img, label

        else:
            raise ValueError("The provided mode is not supported: %s" % self._mode)
    

    def get_pipeline(self):
        """
        Returns a pipeline that was constructed using the parameters specified.
        
        Returns:
        --------
        dataset_images: A tf.data.Dataset pipeline object.
        
        """
        # Create a dataset with records from the CSV file.
        # NOTE: Check the documentation of make_csv_dataset and adjust the
        # parameters to make it more efficient
        list_files = tf.data.experimental.make_csv_dataset(
            self._dataset_file,
            batch_size=32,
            num_epochs=1,
            label_name=self._label_name,
            prefetch_buffer_size=1,
            num_rows_for_inference=100,
            compression_type=None,
            ignore_errors=False,
            **self._kwargs)

        # Parse the data and load the images.
        # NOTE: Check the documentation of map for caching and other optimizations.
        # TODO: Test and support flat-map for "single-all" mode.
        if self._mode == "single-all":
            dataset_images = list_files.flat_map(self._parse_data, num_parallel_calls=self._AUTOTUNE) # TODO: to be tested.
        else:
            dataset_images = list_files.map(self._parse_data,  num_parallel_calls=self._AUTOTUNE)

        # TODO: @Darshan, we need to parse data first in the pipeline before shuffle and repeat and batch operations.
        # TODO: Since doing that with make_csv_dataset is not possible, I think we can use the basic pipeline methods.

        return dataset_images
