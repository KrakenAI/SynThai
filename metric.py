"""
Custom Metric
"""

from collections import OrderedDict

import numpy as np
from sklearn import metrics

import constant

def custom_metric(y_true, y_pred):
    """Calculate score with custom metric"""

    # Sample size and length
    sample_size = y_true.shape[0]
    length = y_true.shape[1]

    # Find score on each metric
    scores = OrderedDict(sorted({
        "seg_accuracy": 0.0,
        "seg_fmeasure": 0.0,
        "pos_accuracy_with_seg": 0.0,
        "pos_accuracy_without_seg": 0.0,
        "pos_fmeasure_with_seg": 0.0,
        "pos_fmeasure_without_seg": 0.0
    }.items()))

    for sample_idx in range(sample_size):
        sample_y_true = y_true[sample_idx]
        sample_y_pred = y_pred[sample_idx]

        # Find segment index
        seg_true_idx = np.argwhere(sample_y_true != constant.NON_SEGMENT_TAG_INDEX)
        seg_pred_idx = np.argwhere(sample_y_pred != constant.NON_SEGMENT_TAG_INDEX)

        # Create segmentation representation in binary array
        seg_true = np.zeros(length)
        seg_true[seg_true_idx] = 1
        seg_pred = np.zeros(length)
        seg_pred[seg_pred_idx] = 1

        # Segmentation accuracy
        scores["seg_accuracy"] += np.mean(np.equal(seg_true, seg_pred))
        scores["seg_fmeasure"] += metrics.f1_score(seg_true, seg_pred,
                                                   pos_label=1, average="binary")

        # POS tagging accuracy
        scores["pos_accuracy_with_seg"] += np.mean(np.equal(sample_y_true, sample_y_pred))

        if len(seg_true_idx) == 0:
            scores["pos_accuracy_without_seg"] += 1
        else:
            scores["pos_accuracy_without_seg"] += np.mean(np.equal(sample_y_true[seg_true_idx],
                                                                   sample_y_pred[seg_true_idx]))

        scores["pos_fmeasure_with_seg"] += metrics.f1_score(sample_y_true,
                                                            sample_y_pred,
                                                            average="weighted")

        if len(seg_true_idx) == 0:
            scores["pos_fmeasure_without_seg"] += 1
        else:
            scores["pos_fmeasure_without_seg"] += metrics.f1_score(sample_y_true[seg_true_idx],
                                                                   sample_y_pred[seg_true_idx],
                                                                   average="weighted")

    # Average score on each metric
    for metric, score in scores.items():
        scores[metric] = score / sample_size

    return scores
