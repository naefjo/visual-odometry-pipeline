import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
from numpy.typing import NDArray

from structures import *


def bootstrap_vo_pipeline(
    img0: cv2.Mat,
    img1: cv2.Mat,
    K: NDArray,
) -> Tuple[
    Tuple[List[cv2.KeyPoint], NDArray], Tuple[NDArray, NDArray], Tuple[int, NDArray]
]:
    """
    Bootstraps the Visual Odometry pipeline,

    Bootstrap the visual odometry pipeline by computing and matching SIFT features
    between the provided images. Using the resulting machtes, the essential matrix is
    estimated and the resulting pose estimate is recovered.

    based on: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

    Args:
        img0: The first image used for bootstrapping the VO pipeline.
        img1: The second image used for bootstrapping the VO pipeline.
        K: Intrinsic camera matrik (3x3)

    Returns:
        keypoints: Tuple containing keypoints in img1 and their corresponding descriptor
        estimated_pose: Tuple of rotation matrix and scaled translation vector which expresses
          the coordinate system of camera 0 in the coordinate system of camera 1.
        inliner_stats: Tuple of number of inliers and inlier mask.

    """

    sift = cv2.SIFT_create()

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
    # matched_keypoints = bf_matcher.match(
    #     keypoint_descriptor_img0, keypoint_descriptor_img1
    # )

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

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matched_keypoints = flann_matcher.knnMatch(
        keypoint_descriptor_img0, keypoint_descriptor_img1, k=2
    )

    # Sort matched keypoints according to keypoint distance
    # matched_keypoints = sorted(matched_keypoints, key=lambda x: x.distance)

    points_img0 = []
    points_img1 = []

    for (m, n) in matched_keypoints:
        if m.distance < 0.8 * n.distance:
            points_img0.append(keypoints_img0[m.queryIdx].pt)
            points_img1.append(keypoints_img1[m.trainIdx].pt)

    points_img0 = np.int32(points_img0)
    points_img1 = np.int32(points_img1)

    essential_matrix, inlier_mask = cv2.findEssentialMat(
        points_img0,
        points_img1,
        K,
        method=cv2.RANSAC,
        prob=0.99,
    )

    # [recovered_rotation, recovered_position] = T_{1,0}, i.e. we convert vectors from the first
    # camera frame into vectors in the second camera frame.
    # NOTE(@naefjo): translation is up to scale/unit length
    num_inliers, recovered_rotation, recovered_position, inlier_mask = cv2.recoverPose(
        essential_matrix,
        points_img0,
        points_img1,
        K,
        inlier_mask,
    )

    return (
        (keypoints_img1, keypoint_descriptor_img1),
        (recovered_rotation, recovered_position),
        (num_inliers, inlier_mask),
    )
