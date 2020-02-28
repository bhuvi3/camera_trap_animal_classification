#!python
# -*- coding: utf-8 -*-

"""
Compute ROC by considering majority voting from the given pred-scores and ground-truth labels.

"""

from sklearn.metrics import auc, confusion_matrix

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats


def get_args():
    parser = argparse.ArgumentParser(description="Compute ROC by considering majority voting from the "
                                                 "given pred-scores and ground-truth labels.")

    parser.add_argument('--preds-labels-file',
                        required=True,
                        help="The path to the pickle file containing the lists of preds and labels.")
    parser.add_argument('--out-file-prefix',
                        required=True,
                        help="The prefix for the path to the to which the ROC curve needs to be written.")
    parser.add_argument('--num-thresh',
                        type=int,
                        default=100,
                        help="The number of thresholds to consider. Default: 100.")

    args = parser.parse_args()
    return args


def _plot_roc(fpr, tpr, out_file, title_suffix):
    roc_auc = auc(fpr, tpr)
    print("The ROC AUC (%s) is: %.2f" % (title_suffix, roc_auc))

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Positive Class: %s' % title_suffix)
    plt.legend(loc="lower right")
    plt.savefig(out_file)
    plt.clf()


def _get_fpr_tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)  # fp / N
    tpr = tp / (tp + fn)  # tp / P
    return fpr, tpr


def compute_roc(preds_labels_file, out_file_prefix, num_thresh=100):
    with open(preds_labels_file, "rb") as fp:
        pred_labels_lists = pickle.load(fp)

    num_entries = len(pred_labels_lists)
    labels = np.array([x[1] for x in pred_labels_lists[0]], dtype=np.int8)  # Pick the labels from the first entry of pred_labels.
    all_scores = []
    for i in range(num_entries):
        all_scores.append([x[0] for x in pred_labels_lists[i]])
        cur_labels = np.array([x[1] for x in pred_labels_lists[i]], dtype=np.int8)
        assert np.allclose(labels, cur_labels)  # Ensures that the labels match across the entries of pred_labels.
    all_scores = np.array(all_scores)

    # Get the thresholds.
    thresholds = np.linspace(-0.01, 1.01, num=num_thresh, endpoint=True)[::-1]

    voted_fpr = []; first_fpr = []
    voted_tpr = []; first_tpr = []

    for thresh in thresholds:
        all_preds = np.array(all_scores >= thresh, dtype=np.int8)

        # Get first fpr and tpr.
        cur_y_pred_first = all_preds[0]
        cur_first_fpr, cur_first_tpr = _get_fpr_tpr(labels, cur_y_pred_first)
        first_fpr.append(cur_first_fpr)
        first_tpr.append(cur_first_tpr)

        # Get voted fpr and tpr.
        cur_y_pred_voted = stats.mode(all_preds).mode.astype(np.int8)[0]
        cur_voted_fpr, cur_voted_tpr = _get_fpr_tpr(labels, cur_y_pred_voted)
        voted_fpr.append(cur_voted_fpr)
        voted_tpr.append(cur_voted_tpr)

    _plot_roc(first_fpr, first_tpr, out_file_prefix + "-first.png", "First Entry")
    _plot_roc(voted_fpr, voted_tpr, out_file_prefix + "-voting.png", "Voted from %s Entries" % num_entries)


if __name__ == "__main__":
    args = get_args()
    compute_roc(args.preds_labels_file, args.out_file_prefix, num_thresh=args.num_thresh)
