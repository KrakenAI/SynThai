import constant
import numpy as np
from collections import OrderedDict
from sklearn import metrics

def custom_metric(x, y_true, y_pred):
    # Sample size and length
    sample_size = x.shape[0]
    length = x.shape[1]

    # Find score on each metric
    scores = OrderedDict({
        "seg_accuracy": 0.0,
        "seg_fmeasure": 0.0,
        "pos_accuracy": 0.0,
        "pos_fmeasure_with_seg": 0.0,
        "pos_fmeasure_without_seg": 0.0
    })

    for sample_idx in range(sample_size):
        # Alias
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

        # Segmentation Accuracy
        scores["seg_accuracy"] += np.mean(np.equal(seg_true, seg_pred))
        scores["seg_fmeasure"] += metrics.f1_score(seg_true, seg_pred,
                                                   pos_label=1, average="binary")

        # POS Tagging Accuracy
        if len(seg_true_idx) == 0:
            scores["pos_accuracy"] += 1
        else:
            scores["pos_accuracy"] += np.mean(np.equal(sample_y_true[seg_true_idx],
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

    # Average score
    for metric, score in scores.items():
        scores[metric] = score / sample_size

    return scores
