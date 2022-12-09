import numpy as np
import cv2
import matplotlib.pyplot as plt

from structures import *

from numpy.typing import NDArray


def bootstrap_vo_pipeline(img0: cv2.Mat, img1: cv2.Mat, K: NDArray) -> None:
    """
    Bootstraps the Visual Odometry pipeline,

    TODO: figure out what inputs/outpus are necessary.

    Proposed bootstrapping pipeline:
    - Run feature extractor on img0 and img1.
    - Find correspondences between extracted correspondences.
    - run PnP algorithm to estimate 3D landmark position of found features.
    - Retrun found landmarks.
    """

    sift = cv2.SIFT_create()

    # create a brute force matcher
    bf_matcher = cv2.BFMatcher_create()

    keypoints_img0, keypoint_descriptor_img0 = sift.detectAndCompute(img0, None)
    keypoints_img1, keypoint_descriptor_img1 = sift.detectAndCompute(img1, None)

    # outimg = None
    # cv2.imshow(
    #     "",
    #     cv2.drawKeypoints(
    #         img0,
    #         keypoints_img0,
    #         outimg,
    #         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #     ),
    # )
    # cv2.waitKey(0)

    # Match SIFT feature descriptors and sort them by decreasing distance.
    matched_keypoints = bf_matcher.match(
        keypoint_descriptor_img0, keypoint_descriptor_img1
    )
    matched_keypoints = sorted(matched_keypoints, key=lambda x: x.distance)

    # Draw matches.
    # img3 = cv2.drawMatches(
    #     img0,
    #     keypoints_img0,
    #     img1,
    #     keypoints_img1,
    #     matched_keypoints[:100],
    #     None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )
    # plt.imshow(img3)
    # plt.show(block=True)
