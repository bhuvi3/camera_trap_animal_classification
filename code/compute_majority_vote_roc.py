#!python
# -*- coding: utf-8 -*-

"""
Compute ROC by considering majority voting from the given pred-scores and ground-truth labels.

"""

from sklearn.metrics import roc_curve, auc, confusion_matrix

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats


def get_args():
    parser = argparse.ArgumentParser(description="Run the the baseline model training.")

    parser.add_argument('--preds-labels-file',
                        required=True,
                        help="The path to the pickle file containing the lists of preds and labels.")
    parser.add_argument('--out-file',
                        required=True,
                        help="The path to the to which the ROC curve needs to be written.")

    args = parser.parse_args()
    return args


def compute_majority_vote_roc(preds_labels_file, out_file):
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

    _, _, thresholds = roc_curve(labels, all_scores[0])

    voted_fpr = []
    voted_tpr = []
    for thresh in thresholds:
        all_preds = all_scores >= thresh
        cur_y_pred = stats.mode(all_preds.astype(np.int8)).mode.astype(np.int8)[0]
        tn, fp, fn, tp = confusion_matrix(labels, cur_y_pred).ravel()
        voted_fpr.append(fp / (fp + tn))  # fp / N
        voted_tpr.append(tp / (tp + fn))  # tp / P

    roc_auc = auc(voted_fpr, voted_tpr)
    print("The Majority Voted ROC AUC is: %.2f" % roc_auc)

    plt.figure()
    plt.plot(voted_fpr, voted_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Positive Class by Majority Voting of %s Entries' % num_entries)
    plt.legend(loc="lower right")
    plt.savefig(out_file)
    plt.clf()


if __name__ == "__main__":
    args = get_args()
    compute_majority_vote_roc(args.preds_labels_file, args.out_file)
