import numpy as np
import pytest
from utils.calibration import recall_threshold

def test_highest_cutoff_meeting_target_and_ties():
    y=[1,1,1,1,0];s=[4,3,3,1,9]
    assert recall_threshold(y,s,.75)==3
    assert recall_threshold(y,s,1)==1

def test_invalid_inputs():
    for y,s,t in [([0],[2],.9),([1],[float('nan')],.9),([1],[1],0),([1,0],[1],.9)]:
        with pytest.raises(ValueError):recall_threshold(y,s,t)

def test_threshold_only_depends_on_calibration_inputs():
    labels=np.array([1,1,0,0]);scores=np.array([2.,1.,3.,0.])
    threshold=recall_threshold(labels,scores,.9)
    scores[labels==0]=100
    assert recall_threshold(labels,scores,.9)==threshold
