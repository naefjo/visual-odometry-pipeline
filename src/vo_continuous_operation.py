import numpy as np
import cv2
import matplotlib.pyplot as plt

from structures import *
from utils import *


def processFrame(
    img,
    K,
    keypoints_prev,
    descriptors_prev,
    landmarks,
    keypoint_detector,
):
    # TODO: detect keypoints in new image

    # TODO: find landmarks corresponding to newly found keypoints by
    # comparing to the keypoints in the previous image.

    # TODO: PnP between landmarks and corresponding image points
