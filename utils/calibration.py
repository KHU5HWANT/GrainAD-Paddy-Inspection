"""Validation-only threshold selection for scores increasing with anomaly."""
import numpy as np

def recall_threshold(labels,scores,target_recall=0.9):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=float)
    if labels.ndim!=1 or scores.shape!=labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected finite, matching one-dimensional labels and scores')
    if not 0<target_recall<=1 or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid target recall or binary labels')
    positive=np.sort(scores[labels==1])[::-1]
    if not len(positive):raise ValueError('Calibration requires defective examples')
    return float(positive[int(np.ceil(target_recall*len(positive)))-1])
