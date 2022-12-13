import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
from numpy.typing import NDArray

from structures import *
from utils import *


def bootstrapVoPipeline(
    img0: cv2.Mat,
    img1: cv2.Mat,
    K: NDArray,
) -> Tuple[Tuple[List[cv2.KeyPoint], NDArray], NDArray, Tuple[int, NDArray]]:
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
        keypoints: Tuple containing keypoints in img1 and their corresponding descriptor.
        landmarks: 4xN array of triangulated landmarks from keypoints between img0 and img1.
        estimated_pose: SE(3) transformation matrix which expresses
          the coordinate system of camera 1 in the coordinate system of camera 0.
          NOTE: translation is only up to scale
        inliner_stats: Tuple of number of inliers and inlier mask.

    """

    sift = cv2.SIFT_create()

    keypoints_img0, keypoint_descriptor_img0 = sift.detectAndCompute(img0, None)
    keypoints_img1, keypoint_descriptor_img1 = sift.detectAndCompute(img1, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = (
        cv2.FlannBasedMatcher_create()
    )  # FlannBasedMatcher(index_params, search_params)
    matched_keypoints = flann_matcher.knnMatch(
        keypoint_descriptor_img0, keypoint_descriptor_img1, k=2
    )

    good_matches = []
    # arranged according to [Nx2]
    points_img0 = []
    points_img1 = []

    for (match_1, match_2) in matched_keypoints:
        if match_1.distance < 0.8 * match_2.distance:
            good_matches.append(match_1)
            points_img0.append(keypoints_img0[match_1.queryIdx].pt)
            points_img1.append(keypoints_img1[match_1.trainIdx].pt)

    points_img0 = np.int32(points_img0)
    points_img1 = np.int32(points_img1)

    essential_matrix, inlier_mask = cv2.findEssentialMat(
        points_img0,
        points_img1,
        K,
        method=cv2.RANSAC,
        prob=0.99,
    )

    drawImagesWithCorrespondingKeypoints(
        img0,
        img1,
        keypoints_img0,
        keypoints_img1,
        good_matches,
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

    T_img1_img0 = np.eye(4)
    T_img1_img0[:3, :3] = recovered_rotation
    T_img1_img0[:3, -1:] = recovered_position.reshape(3, 1)
    # T_img0_img1 = np.linalg.inv(T_img1_img0)
    T_img0_img1 = invertSE3Matrix(T_img1_img0)

    projection_matrix_0 = (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(float)
    projection_matrix_1 = (K @ (T_img1_img0[:3, :])).astype(float)
    projection_points_0 = (points_img0.T).astype(float)
    projection_points_1 = (points_img1.T).astype(float)

    landmarks = cv2.triangulatePoints(
        projMatr1=projection_matrix_0,
        projMatr2=projection_matrix_1,
        projPoints1=projection_points_0,
        projPoints2=projection_points_1,
    )

    # Normalize last coordinate?
    # TODO(@naefjo): figure out if this is correct or no.
    landmarks /= landmarks[3, :]
    landmark_distance_mask = np.linalg.norm(landmarks, axis=0) < 75
    landmark_z_direction_mask = landmarks[2, :] > 0
    landmarks = landmarks[:, landmark_distance_mask & landmark_z_direction_mask]

    return (
        (keypoints_img1, keypoint_descriptor_img1),
        landmarks,
        T_img0_img1,
        (num_inliers, inlier_mask),
    )
