#!python
# -*- coding: utf-8 -*-

"""
Run model inference and  evaluation on the given test set.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from data_pipeline import PipelineGenerator
from models import ModelFactory
from scipy import stats
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Callable, Optional

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import tensorflow as tf
import time


def get_args():
    parser = argparse.ArgumentParser(description="Run model inference and  evaluation on the given test set.")

    parser.add_argument('--test-meta-file',
                        required=True,
                        help="The path to the file containing the test images metadata.")
    parser.add_argument('--images-dir',
                        required=True,
                        help="The path to the directory containing the images.")
    parser.add_argument('--out-dir',
                        required=True,
                        help="The path to the output dir to which the evaluation results files need to be written.")
    parser.add_argument('--data-pipeline-mode',
                        required=True,
                        help="The mode to be used for the data pipeline.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help="The number of datapoints in a training batch. This size must fit within the GPU memory. "
                             "Default: 64.")
    parser.add_argument('--trained-model-arch',
                        required=True,
                        help="The name of the trained model architecture which is present in the 'models' module.")
    parser.add_argument('--trained-checkpoint-dir',
                        required=True,
                        help="The name of the model checkpoint directory created using train pipeline.")
    parser.add_argument('--filetype',
                        default="tf",
                        help="The type of saved model used, select from ['.h5', 'tf']. Default: 'tf'.")
    parser.add_argument('--extract-layers',
                        default=None,
                        help="The names of the model layers (as a comma separated values) to be extracted. "
                             "Default: None.")
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

    if not args.extract_layers:
        args.extract_layers = []
    else:
        args.extract_layers = args.extract_layers.split(",")

    return args


# Taken from n2cholas, from link: https://github.com/tensorflow/tensorflow/issues/33478.
class LayerWithHooks(tf.keras.layers.Layer):
    def __init__(self, layer: tf.keras.layers.Layer,
                 hooks: List[Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]] = None):
        super().__init__()
        self._layer = layer
        self._hooks = hooks or []

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self._layer(input)
        for hook in self._hooks:
            hook_result = hook(input, output)
            if hook_result is not None:
                output = hook_result
        return output

    def register_hook(self, hook: Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]) -> None:
        self._hooks.append(hook)


class InputOutputSaver:
    def __call__(self, input: tf.Tensor, output: tf.Tensor) -> None:
        self.input = input
        self.output = output


def get_layer_outputs(model, required_layers, input_tensors):
    """
    Extract the outputs of intermediate layers provided in required_layers from the model,
    after running the input_batch through the model.
    Returns a dictionary of 'savers'. Access a particular layer output as followed:
        savers[layer_name].output

    """
    def _get_call_fn(layer: tf.keras.layers.Layer) -> Callable[[tf.Tensor], tf.Tensor]:
        old_call_fn = layer.call

        def call(input: tf.Tensor) -> tf.Tensor:
            output = old_call_fn(input)
            for hook in layer._hooks:
                hook_result = hook(input, output)
                if hook_result is not None:
                    output = hook_result
            return output

        return call

    # Register hooks and savers.
    savers = {}
    for layer_name in required_layers:
        layer = model.get_layer(layer_name)
        layer._hooks = []
        layer.call = _get_call_fn(layer)
        layer.register_hook = lambda hook: layer._hooks.append(hook)
        saver = InputOutputSaver()
        layer.register_hook(saver)
        savers[layer_name] = saver

    model_outputs = model(input_tensors, training=False)
    savers["output"] = model_outputs
    return savers


def load_and_get_model_for_inference(trained_model_arch, trained_checkpoint_dir, filetype, input_shape, num_classes):
    model_factory = ModelFactory()
    model = model_factory.get_model(trained_model_arch,
                                    input_shape,
                                    is_training=False,
                                    num_classes=num_classes,
                                    learning_rate=0.001)  # A dummy learning rate since it is test mode.
    # The ModelCheckpoint in train pipeline saves the weights inside the checkpoint directory as follows.
    if filetype == '.h5':
        weights_path = trained_checkpoint_dir + "best_model_dir-auc.h5"
        model = tf.keras.models.load_model(weights_path)
    elif filetype == 'tf':
        weights_path = os.path.join(trained_checkpoint_dir, "variables", "variables")
        model.load_weights(weights_path)
    else:
        raise ValueError("The provided saved model filetype not recognized: %s" % filetype)

    print("The model has been created and the weights have been loaded from: %s" % weights_path)
    model.summary()
    return model


def evaluate(y_pred, evaluation_dir):
    """Get evaluation metrics on the given pred_labels list.
    """
    print("Currently only ROC is supported. AP over recall range of 90:100 would be included!")
    os.makedirs(evaluation_dir)
    y_score = [x[0] for x in y_pred]
    y_test = [x[1] for x in y_pred]

    # Compute ROC score.
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("The ROC AUC is: %.2f" % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Positive Class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(evaluation_dir, "roc.png"))
    plt.clf()

    # Compute score distribution.
    score_stats = stats.describe(y_score)
    print("Predicted Probabilities Distribution: %s" % str(score_stats))
    plt.style.use('ggplot')
    plt.hist(y_score, bins=100, color="blue")
    plt.title("Score Distribution")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(evaluation_dir, "score_distribution.png"))
    plt.clf()

    # Applying mean as threshold and providing other useful metrics.
    thresh = score_stats.mean
    print("Applying threshold (mean): %s (Please check the ROC curve and apply appropriate threshold)" % thresh)
    y_pred = y_score >= thresh  # APplying threshold to get predictions.
    prediction_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "preciion": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1-score": f1_score(y_test, y_pred)
    }
    print("Other Performance Metrics (after applying mean as threshold): \n%s" % prediction_metrics)

    print("Evaluation results have been written to: %s" % evaluation_dir)


def inference_pipeline(test_metadata_file_path,
                       images_dir_path,
                       out_dir,
                       trained_model_arch,
                       trained_checkpoint_dir,
                       filetype,
                       num_classes,
                       label_name=None,
                       sequence_image_count=3,
                       data_pipeline_mode="mode_flat_all",
                       batch_size=32,
                       input_size=(224, 224, 3),
                       extract_layers=None):
    """
    Evaluates the provided trained model on the given test set. If it is not a sequence model
    (i.e., a single image model), then it returns both first-image and max-probability results.

    :param test_metadata_file_path: The path to the file containing the test images metadata.
    :param images_dir_path: The path to the directory containing the images.
    :param out_dir: The path to the output dir to which the evaluation results files need to be written.
    :param trained_model_arch: The name of the trained model architecture which is present in the 'models' module.
    :param trained_checkpoint_dir: The name of the model checkpoint directory created using train pipeline.
    :param filetype: The type of saved model used, select from ['.h5', 'tf']. Default: 'tf'.
    :param num_classes: The number of classes.
    :param label_name: The name of the label column in the metadata csv file.
    :param sequence_image_count: The number of images in the sequence dataset. Default: 3.
    :param is_sequence_model: Specify True if this is a sequence-based model. Default False.
    :param batch_size: The batch size used for the data. Ensure that it fits within the GPU memory. Default: 32.
    :param input_size: The shape of the tensors returned by the data pipeline mode. Default: (224, 224, 3).
    :param extract_layers: The names of the model layers (as a comma separated values) to be extracted.

    """
    def _get_preds_labels(model, test_input_tensors, test_labels):
        preds = model.predict(test_input_tensors)
        preds_labels = list(zip(preds[:, 0], test_labels.numpy()))
        return preds_labels

    def _get_max_proba_pred_labels(pred_labels):
        all_preds = []
        all_labels = []
        for i, cur_pred_labels in enumerate(pred_labels):
            # Each individual image pred_labels list.
            all_preds.append([x[0] for x in cur_pred_labels])
            all_labels.append([x[1] for x in cur_pred_labels])

            # Ensure all labels match.
            # TODO: This step is redundant, it can be removed.
            if i > 0:
                assert np.allclose(all_labels[0], all_labels[i])

        # Get max probas.
        max_pred_probas = np.max(np.vstack(all_preds), axis=0)
        max_proba_pred_labels = list(zip(max_pred_probas, all_labels[0]))
        return max_proba_pred_labels

    # Create the evaluation output directory, and save a copy of the test_metadata_file being evaluated.
    os.makedirs(out_dir)
    shutil.copy(test_metadata_file_path, os.path.join(out_dir, os.path.basename(test_metadata_file_path)))

    # Infer if the architecture works with sequence of images
    is_sequence_model = data_pipeline_mode not in PipelineGenerator.SINGLE_IMAGE_MODES

    # Load the test data pipeline.
    if is_sequence_model:  # For all models which use sequence information in some way, use respective modes.
        pipeline_mode = data_pipeline_mode
    else:  # For single-image models, use the MODE_SEQUENCE, and infer on each individual image.
        pipeline_mode = PipelineGenerator.MODE_SEQUENCE

    pipeline_gen = PipelineGenerator(test_metadata_file_path,
                                     images_dir_path,
                                     is_training=False,
                                     sequence_image_count=sequence_image_count,
                                     label_name=label_name,
                                     mode=pipeline_mode)  # We always use sequence level inference during evaluation.

    test_dataset_raw = pipeline_gen.get_pipeline()
    test_dataset_batches = test_dataset_raw.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    num_test_sequences = pipeline_gen.get_size()
    print("There are %s test sequences." % num_test_sequences)

    # Create the model architecture and load the trained weights from checkpoint dir.
    model = load_and_get_model_for_inference(trained_model_arch, trained_checkpoint_dir, filetype, input_size, num_classes)

    # Extract the predicted probabilities and corresponding labels.
    # This would be a list of (pred, label) tuples if this is_sequence_model.
    # Else, this list would contain, 'sequence_image_count' (say 3) lists of (pred, label) tuples.
    print("\nRunning model inference.")
    start_time = time.time()
    if is_sequence_model:
        pred_labels = []
    else:  # If single-image based models are running, run inference on each image.
        pred_labels = [[] for _ in range(sequence_image_count)]

    for test_batch in test_dataset_batches:
        test_sequences, test_labels = test_batch
        if is_sequence_model:
            cur_pred_labels = _get_preds_labels(model, test_sequences, test_labels)
            pred_labels.extend(cur_pred_labels)
        else:
            for i in range(sequence_image_count):
                cur_single_image_test_images = test_sequences[:, i, :, :, :]
                cur_single_image_pred_labels = _get_preds_labels(model, cur_single_image_test_images, test_labels)
                pred_labels[i].extend(cur_single_image_pred_labels)

    end_time = time.time()
    time_taken = end_time - start_time
    print("Running inference on the test set has completed in %s seconds on the test set metadata file: %s"
          % (time_taken, test_metadata_file_path))

    print("Asserting the number of pred_labels found.")
    if is_sequence_model:
        assert len(pred_labels) == num_test_sequences
    else:
        assert len(pred_labels[0]) == num_test_sequences

    # Save the pred labels to a pickle file.
    if is_sequence_model:
        pred_label_output_path = os.path.join(out_dir, "pred_labels-sequence.pickle")
    else:  # If single-image based models are running, run inference on each image.
        pred_label_output_path = os.path.join(out_dir, "pred_labels-individual.pickle")
    with open(pred_label_output_path, "wb") as fp:
        pickle.dump(pred_labels, fp)
    print("Model inference has completed, and the pred_labels have been saved to: %s", pred_label_output_path)

    # Run evaluations.
    print("\nRunning evaluations.")
    evaluation_dir = os.path.join(out_dir, "evaluation")
    if is_sequence_model:
        evaluate(pred_labels, os.path.join(evaluation_dir, "sequence_model"))
    else:
        # For single-image models, run two evaluations, one for the first image only, then for the max-probability.
        evaluate(pred_labels[0], os.path.join(evaluation_dir, "individual-first_image"))

        max_pred_labels = _get_max_proba_pred_labels(pred_labels)
        evaluate(max_pred_labels, os.path.join(evaluation_dir, "individual-max_proba"))

    # Save the layer outputs for the extract_layers to a pickle file.
    if extract_layers:
        print("\nExtracting the outputs at intermediate layers.")
        savers_outpath = os.path.join(out_dir, "extracted_layer_outputs_dict.pickle")
        if is_sequence_model:
            savers = get_layer_outputs(model, extract_layers, test_sequences)
        else:
            # Consider the first image for individual image models.
            savers = get_layer_outputs(model, extract_layers, test_sequences[:, 0, :, :])
        with open(savers_outpath, "wb") as fp:
            pickle.dump(savers, fp)
        print("The outputs at these model layers - %s - from the last batch have been saved to: %s"
              % (extract_layers, savers_outpath))

    print("Inference and evaluation has been completed and outputs can be found in: %s" % out_dir)


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

    inference_pipeline(args.test_meta_file,
                       args.images_dir,
                       args.out_dir,
                       args.trained_model_arch,
                       args.trained_checkpoint_dir,
                       args.filetype,
                       num_classes,
                       label_name=label_name,
                       sequence_image_count=args.sequence_image_count,
                       data_pipeline_mode=args.data_pipeline_mode,
                       batch_size=args.batch_size,
                       input_size=input_size,
                       extract_layers=args.extract_layers)
