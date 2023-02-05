import numpy as np
import cv2
import matplotlib.pyplot as plt

from structures import *
from utils import *
from feature_detector import SIFTKeypointDetectorAndMatcher

DEBUG_PLOTS = True


def processFrame(
    img_prev: cv2.Mat,
    img_new: cv2.Mat,
    K: NDArray,
    previous_image_features: ImageFeatures,
    feature_detector: SIFTKeypointDetectorAndMatcher,
) -> ImageFeatures:
    """
    Frame processing section of the VO continuous operation.

    Processes the incoming frames and performs the PnP algorithm to localize new frames.

    Args:
        img_new: New image which needs to be processed.
        K: Camera intrinsics matrix
        keypoints_prev_image: Keypoints which were found in the previous image.
        descriptors_prev_image: Descriptors of the keypoints found in the previous image.
        landmarks_I: landmarks, in the inertial frame, which correspond to the keypoints from
          the previous image. I.e. landmarks_I[i] corresponds to keypoints_prev_image[i] and
          descriptors_prev_image[i].
        T_I_prev: Pose of the previous camera in the inertial frame.
        feature_detector: feature detector object which does the feature detection and matching
          between the frames.
    """
    # TODO: detect keypoints in new image
    keypoints_new_image, descriptors_new_image = feature_detector.detect_keypoints(
        img_new
    )

    # TODO: find landmarks corresponding to newly found keypoints by
    # comparing to the keypoints in the previous image.
    (
        matches,
        matched_keypoints_new_image,
        matched_descriptors_new_image,
        matched_landmarks,
    ) = feature_detector.MatchKeypointsAndFindLandmarks(
        previous_image_features.keypoints[1],
        descriptors_new_image,
        keypoints_new_image,
        previous_image_features.landmarks,
    )

    kp_coords_new_image = feature_detector.getImageCoordinatesArray(
        matched_keypoints_new_image
    )

    if DEBUG_PLOTS:
        drawImagesWithCorrespondingKeypoints(
            img_prev,
            img_new,
            previous_image_features.keypoints[0],
            keypoints_new_image,
            matches,
        )

    # TODO: PnP between landmarks and corresponding image points
    pnp_params = {
        "objectPoints": matched_landmarks[:3, :].T,
        "imagePoints": kp_coords_new_image,
        "cameraMatrix": K,
        "distCoeffs": None,
        "flags": cv2.SOLVEPNP_P3P,
        "reprojectionError": 0.3,
        "confidence": 0.999,
        # "rvec": cv2.Rodrigues(T_I_prev[:3, :3]),
        # "tvec": T_I_prev[:3, -1],
    }
    try:
        pnp_success, rotation_vec, translation_vec, inlier_history = cv2.solvePnPRansac(
            **pnp_params
        )
    except:
        pnp_success = False

    if pnp_success:
        T_new_I = computeHomogeneousTransformationMatrix(
            cv2.Rodrigues(rotation_vec)[0], translation_vec
        )
        T_I_new = invertSE3Matrix(T_new_I)
    else:
        print("pnp crashed and burned. hoorayyy")
        return previous_image_features

    return ImageFeatures(
        (matched_keypoints_new_image, matched_descriptors_new_image),
        matched_landmarks,
        T_I_new,
        (0, 0),  # TODO: fix inlier stats
        (keypoints_new_image, descriptors_new_image),
    )


def localizeNewLandmarks(
    K: NDArray,
    keyframe_img,
    keyframe_image_features: ImageFeatures,
    curr_img,
    previous_image_features: ImageFeatures,
    feature_detector: SIFTKeypointDetectorAndMatcher,
) -> ImageFeatures:

    (
        matches,
        matched_keypoints_image_0,
        matched_keypoints_image_1,
        matched_keypoint_descriptors_image_1,
    ) = feature_detector.MatchKeypointsAndDescriptors(
        keyframe_image_features.all_keypoints[0],
        previous_image_features.all_keypoints[0],
        keyframe_image_features.all_keypoints[1],
        previous_image_features.all_keypoints[1],
    )

    drawImagesWithCorrespondingKeypoints(
        keyframe_img,
        curr_img,
        keyframe_image_features.all_keypoints[0],
        previous_image_features.all_keypoints[0],
        matches,
    )

    kp_coords_img0 = feature_detector.getImageCoordinatesArray(
        matched_keypoints_image_0
    )
    kp_coords_img1 = feature_detector.getImageCoordinatesArray(
        matched_keypoints_image_1
    )

    # Run ransac on the found features to remove outliers
    essential_matrix, inlier_mask = cv2.findEssentialMat(
        kp_coords_img0,
        kp_coords_img1,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    kp_coords_img0 = kp_coords_img0[inlier_mask, :]
    kp_coords_img1 = kp_coords_img1[inlier_mask, :]
    matched_keypoints_image_1 = matched_keypoints_image_1[inlier_mask]
    matched_keypoint_descriptors_image_1 = matched_keypoint_descriptors_image_1[
        inlier_mask, :
    ]

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
    matched_keypoints_image_1 = matched_keypoints_image_1[inlier_mask]
    matched_keypoint_descriptors_image_1 = matched_keypoint_descriptors_image_1[
        inlier_mask, :
    ]

    drawImagesWithCorrespondingKeypoints(
        keyframe_img,
        curr_img,
        keyframe_image_features.all_keypoints[0],
        previous_image_features.all_keypoints[0],
        matches,
    )

    # Construct homogeneous transformation of estimated pose
    T_img1_img0 = computeHomogeneousTransformationMatrix(
        recovered_rotation, recovered_position
    )

    # triangulation_parameters = {
    #     "projMatr1": (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(float),
    #     "projMatr2": (K @ (T_img1_img0[:3, :])).astype(float),
    #     "projPoints1": (kp_coords_img0.T).astype(float),
    #     "projPoints2": (kp_coords_img1.T).astype(float),
    # }

    # landmarks = cv2.triangulatePoints(**triangulation_parameters)
    # landmarks /= landmarks[3, :]
    # landmarks = keyframe_image_features.transform @ landmarks

    triangulation_parameters = {
        "projMatr1": (
            K @ keyframe_image_features.get_transform_world_to_camera_3x4()
        ).astype(float),
        "projMatr2": (
            K @ previous_image_features.get_transform_world_to_camera_3x4()
        ).astype(float),
        "projPoints1": (kp_coords_img0.T).astype(float),
        "projPoints2": (kp_coords_img1.T).astype(float),
    }
    landmarks = cv2.triangulatePoints(**triangulation_parameters)
    landmarks /= landmarks[3, :]

    previous_image_features.landmarks = landmarks
    previous_image_features.keypoints = (
        matched_keypoints_image_1,
        matched_keypoint_descriptors_image_1,
    )
    return previous_image_features
