import cv2
import numpy as np


class SIFTKeypointDetectorAndMatcher:
    def __init__(self, sift_params):
        self.sift_detector = cv2.SIFT_create(**sift_params)
        self.flann_matcher = cv2.FlannBasedMatcher_create()

    def detect_keypoints(self, img):
        return self.sift_detector.detectAndCompute(img, None)

    def match_keypoints(self, descriptor_0, descriptor_1):
        return self.flann_matcher.knnMatch(descriptor_0, descriptor_1, k=2)

    def MatchKeypointsAndDescriptors(
        self, keypoints_0, keypoints_1, descriptors_0, descriptors_1
    ):
        """
        Match features between image 0 and image 1 given the keypoints and descriptors.

        This method is used during the bootstrapping process of the pipeline.
        """
        matches = self.match_keypoints(descriptors_0, descriptors_1)

        actual_matches = []
        # arranged according to [Nx2]
        matched_keypoints_image_0 = []
        matched_keypoints_image_1 = []
        matched_keypoint_descriptors_image_1 = []

        for (match_1, match_2) in matches:
            if match_1.distance < 0.8 * match_2.distance:
                actual_matches.append(match_1)
                matched_keypoints_image_0.append(keypoints_0[match_1.queryIdx])
                matched_keypoints_image_1.append(keypoints_1[match_1.trainIdx])
                matched_keypoint_descriptors_image_1.append(
                    descriptors_1[match_1.trainIdx]
                )

        matched_keypoints_image_0 = np.array(matched_keypoints_image_0, dtype=object)
        matched_keypoints_image_1 = np.array(matched_keypoints_image_1, dtype=object)
        matched_keypoint_descriptors_image_1 = np.array(
            matched_keypoint_descriptors_image_1
        )

        return (
            actual_matches,
            matched_keypoints_image_0,
            matched_keypoints_image_1,
            matched_keypoint_descriptors_image_1,
        )

    def MatchKeypointsAndFindLandmarks(
        self, descriptors_0, descriptors_1, keypoints_1, landmarks
    ):
        """
        Match features between images and find landmarks which correspond to features in image 1.

        This method is used in the continuous operation phase of the pipeline.
        """
        matches = self.match_keypoints(descriptors_0, descriptors_1)

        actual_matches = []
        matched_keypoints_image_1 = []
        matched_keypoint_descriptors_image_1 = []
        matched_landmarks = []

        for (match_1, match_2) in matches:
            if match_1.distance < 0.8 * match_2.distance:
                actual_matches.append(match_1)
                matched_keypoints_image_1.append(keypoints_1[match_1.trainIdx])
                matched_keypoint_descriptors_image_1.append(
                    descriptors_1[match_1.trainIdx]
                )
                matched_landmarks.append(landmarks[:, match_1.queryIdx].T)

        return (
            actual_matches,
            matched_keypoints_image_1,
            np.array(matched_keypoint_descriptors_image_1),
            np.array(matched_landmarks).T,
        )

    @staticmethod
    def getImageCoordinatesArray(keypoints):
        image_coordinates = []
        for keypoint in keypoints:
            image_coordinates.append(keypoint.pt)

        return np.array(image_coordinates)
