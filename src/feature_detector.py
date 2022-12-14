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
        matches = self.match_keypoints(descriptors_0, descriptors_1)

        acutal_matches = []
        # arranged according to [Nx2]
        matched_keypoints_image_0 = []
        matched_keypoints_image_1 = []
        matched_keypoint_descriptors_image_1 = []

        for (match_1, match_2) in matches:
            if match_1.distance < 0.8 * match_2.distance:
                acutal_matches.append(match_1)
                matched_keypoints_image_0.append(keypoints_0[match_1.queryIdx])
                matched_keypoints_image_1.append(keypoints_1[match_1.trainIdx])
                matched_keypoint_descriptors_image_1.append(
                    descriptors_1[match_1.trainIdx]
                )

        matched_keypoint_descriptors_image_1 = np.array(
            matched_keypoint_descriptors_image_1
        )

        return (
            acutal_matches,
            matched_keypoints_image_0,
            matched_keypoints_image_1,
            matched_keypoint_descriptors_image_1,
        )

    @staticmethod
    def getImageCoordinatesArray(keypoints):
        image_coordinates = []
        for keypoint in keypoints:
            image_coordinates.append(keypoint.pt)

        return np.array(image_coordinates)
