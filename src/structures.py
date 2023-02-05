from enum import Enum
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional
from numpy.typing import NDArray
from cv2 import KeyPoint
import numpy as np


class DataSet(Enum):
    KITTI = 0
    MALAGA = 1
    PARKING = 2


data_path = {
    DataSet.KITTI: "data/kitti/",
    DataSet.MALAGA: "data/malaga/",
    DataSet.PARKING: "data/parking/",
}

kLandmarkThreshold = 40


@dataclass
class ImageFeatures:
    keypoints: Tuple[List[KeyPoint], NDArray]
    """Tuple containing keypoints in img1 and their corresponding descriptor."""
    landmarks: NDArray
    """4xN array of triangulated landmarks from keypoints between img0 and img1.
        Landmark at index i corresponds to keypoint at index i."""
    transform: NDArray
    """SE(3) transformation matrix which expresses
        the coordinate system of the camera in the world coordinate system. i.e. T_I_C"""
    inlier_stats: Tuple[int, NDArray]
    """Tuple of number of inliers and inlier mask."""
    all_keypoints: Any
    "Tuple containing all detected keypoints of the image and the corresponding descriptors."

    def change_transform(
        self, transform_to_image_frame: Optional[NDArray] = None
    ) -> None:
        if transform_to_image_frame is None:
            transform_to_image_frame = generateWorldFrame()

        self.transform = transform_to_image_frame @ self.transform
        self.landmarks = transform_to_image_frame @ self.landmarks

    def get_transform_world_to_camera_3x4(self) -> NDArray:
        transform_world_to_image = invertSE3Matrix(self.transform)
        return transform_world_to_image[:3, :]


def computeHomogeneousTransformationMatrix(rotation, translation):
    T_img1_img0 = np.eye(4)
    T_img1_img0[:3, :3] = rotation
    T_img1_img0[:3, -1:] = translation.reshape(3, 1)
    return T_img1_img0


def invertSE3Matrix(transformation_matrix: NDArray) -> NDArray:
    """
    Invert a homogeneous transformation matrix belonging to SE(3).
    """
    rotation_matrix = transformation_matrix[:3, :3]
    inverse_rotation_matrix = rotation_matrix.T
    translation_vector = transformation_matrix[:3, -1:]

    inverse_transform = np.eye(4)
    inverse_transform[:3, :3] = inverse_rotation_matrix
    inverse_transform[:3, -1:] = -inverse_rotation_matrix @ translation_vector
    return inverse_transform


def RxToHomogeneousTransform(angle):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def RyToHomogeneousTransform(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def RzToHomogeneousTransform(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def generateWorldFrame() -> NDArray:
    T_I_c0 = np.eye(4)
    T_I_c0[:3, -1:] = np.array([0, 0, 1]).reshape(3, 1)
    T_I_c0 = T_I_c0 @ RxToHomogeneousTransform(-np.pi / 2)
    return T_I_c0
