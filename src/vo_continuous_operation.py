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
    keypoints_prev_image,
    descriptors_prev_image: NDArray,
    landmarks_I: NDArray,
    T_I_prev: NDArray,
    feature_detector: SIFTKeypointDetectorAndMatcher,
):
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
        descriptors_prev_image, descriptors_new_image, keypoints_new_image, landmarks_I
    )

    kp_coords_new_image = feature_detector.getImageCoordinatesArray(
        matched_keypoints_new_image
    )

    if DEBUG_PLOTS:
        drawImagesWithCorrespondingKeypoints(
            img_prev,
            img_new,
            keypoints_prev_image,
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
    pnp_success, rotation_vec, translation_vec, inlier_history = cv2.solvePnPRansac(
        **pnp_params
    )

    if pnp_success:
        T_new_I = computeHomogeneousTransformationMatrix(
            cv2.Rodrigues(rotation_vec)[0], translation_vec
        )
        T_I_new = invertSE3Matrix(T_new_I)
    else:
        print("pnp crashed and burned. hoorayyy")

    print("blbi")
    return (
        (matched_keypoints_new_image, matched_descriptors_new_image),
        matched_landmarks,
        T_I_new,
        (0, 0),  # TODO: fix inlier stats
    )



def localizeNewLandmarks():
    raise NotImplementedError
