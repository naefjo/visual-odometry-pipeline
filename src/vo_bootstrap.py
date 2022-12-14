import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
from numpy.typing import NDArray

from structures import *
from utils import *
from feature_detector import SIFTKeypointDetectorAndMatcher

DEBUG_PLOTS = False


def bootstrapVoPipeline(
    img0: cv2.Mat,
    img1: cv2.Mat,
    K: NDArray,
    feature_detector: SIFTKeypointDetectorAndMatcher,
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

    keypoints_img0, keypoint_descriptor_img0 = feature_detector.detect_keypoints(img0)
    keypoints_img1, keypoint_descriptor_img1 = feature_detector.detect_keypoints(img1)

    (
        matches,
        matched_keypoints_image_0,
        matched_keypoints_image_1,
        matched_keypoint_descriptors_image_1,
    ) = feature_detector.MatchKeypointsAndDescriptors(
        keypoints_img0,
        keypoints_img1,
        keypoint_descriptor_img0,
        keypoint_descriptor_img1,
    )

    kp_coords_img0 = feature_detector.getImageCoordinatesArray(
        matched_keypoints_image_0
    )
    kp_coords_img1 = feature_detector.getImageCoordinatesArray(
        matched_keypoints_image_1
    )

    if DEBUG_PLOTS:
        drawImagesWithCorrespondingKeypoints(
            img0,
            img1,
            keypoints_img0,
            keypoints_img1,
            matches,
        )

    essential_matrix, inlier_mask = cv2.findEssentialMat(
        kp_coords_img0,
        kp_coords_img1,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.3,
    )
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    kp_coords_img0 = kp_coords_img0[inlier_mask, :]
    kp_coords_img1 = kp_coords_img1[inlier_mask, :]
    # matched_keypoints_image_1 = matched_keypoints_image_1[inlier_mask]
    matched_keypoint_descriptors_image_1 = matched_keypoint_descriptors_image_1[
        inlier_mask, :
    ]

    # [recovered_rotation, recovered_position] = T_{1,0}, i.e. we convert vectors from the first
    # camera frame into vectors in the second camera frame.
    # NOTE(@naefjo): translation is up to scale/unit length
    num_inliers, recovered_rotation, recovered_position, inlier_mask = cv2.recoverPose(
        essential_matrix,
        kp_coords_img0,
        kp_coords_img1,
        K,
    )

    # TODO: inlier mask of recover pose ahs only entries 255....
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    kp_coords_img0 = kp_coords_img0[inlier_mask, :]
    kp_coords_img1 = kp_coords_img1[inlier_mask, :]
    # matched_keypoints_image_1 = matched_keypoints_image_1[inlier_mask]
    matched_keypoint_descriptors_image_1 = matched_keypoint_descriptors_image_1[
        inlier_mask, :
    ]

    # Construct homogeneous transformation of estimated pose
    T_img1_img0 = computeTransformation(recovered_rotation, recovered_position)
    T_img0_img1 = invertSE3Matrix(T_img1_img0)

    triangulation_parameters = {
        "projMatr1": (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(float),
        "projMatr2": (K @ (T_img1_img0[:3, :])).astype(float),
        "projPoints1": (kp_coords_img0.T).astype(float),
        "projPoints2": (kp_coords_img1.T).astype(float),
    }

    landmarks = cv2.triangulatePoints(**triangulation_parameters)

    # Normalize last coordinate?
    # TODO(@naefjo): figure out if this is correct or no.
    landmarks /= landmarks[3, :]

    # filter out points which are beyond a certain threshold.
    # landmark_distance_mask = np.linalg.norm(landmarks, axis=0) < 75
    # sum_masked_distance = np.sum(~landmark_distance_mask)
    # landmarks = landmarks[:, landmark_distance_mask]

    # Filter out points which are behind the camera.
    # landmark_z_direction_mask = landmarks[2, :] > 0
    # sum_masked_landmarks_z = np.sum(~landmark_z_direction_mask)
    # landmarks = landmarks[:, landmark_z_direction_mask]

    return (
        (kp_coords_img1, matched_keypoint_descriptors_image_1),
        landmarks,
        T_img0_img1,
        (num_inliers, inlier_mask),
    )


def computeTransformation(rotation, translation):
    T_img1_img0 = np.eye(4)
    T_img1_img0[:3, :3] = rotation
    T_img1_img0[:3, -1:] = translation.reshape(3, 1)
    return T_img1_img0
