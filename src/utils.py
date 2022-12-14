import cv2
import numpy as np

from numpy.typing import NDArray

from structures import data_path, DataSet


def getImagePath(image_index: int, dataset: DataSet) -> str:
    """
    Returns the image path for a given image index and dataset.
    """
    return data_path[dataset] + "images/img_" + str(image_index).zfill(5) + ".png"


def getContinuousOperationImage(image_index, dataset):
    if dataset == DataSet.KITTI:
        raise NotImplementedError
        # image = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);

    if dataset == DataSet.MALAGA:
        raise NotImplementedError
        # image = rgb2gray(imread([malaga_path ...
        #     '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        #     left_images(i).name]));

    if dataset == DataSet.PARKING:
        return cv2.imread(getImagePath(image_index, dataset), cv2.IMREAD_GRAYSCALE)

    raise NotImplementedError


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


def drawImageWithKeypoints(img, keypoints):
    """
    Display an image and the corresponding keypoints.
    """
    cv2.imshow(
        "",
        cv2.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        ),
    )
    cv2.waitKey(0)


def drawImagesWithCorrespondingKeypoints(img0, img1, kp_0, kp_1, matched_keypoints):
    """
    show images and the keypoints which have been matched between the images.
    """
    cv2.imshow(
        "",
        cv2.drawMatches(
            img0,
            kp_0,
            img1,
            kp_1,
            matched_keypoints,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        ),
    )
    cv2.waitKey(0)


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


def plotCoordinateSystemFromTransform(transform, ax, scale=1):
    start = transform[:3, -1]
    x_unit_vec = transform[:3, 0]
    y_unit_vec = transform[:3, 1]
    z_unit_vec = transform[:3, 2]

    ax.plot(
        [start[0], start[0] + scale * x_unit_vec[0]],
        [start[1], start[1] + scale * x_unit_vec[1]],
        [start[2], start[2] + scale * x_unit_vec[2]],
        "r",
    )
    ax.plot(
        [start[0], start[0] + scale * y_unit_vec[0]],
        [start[1], start[1] + scale * y_unit_vec[1]],
        [start[2], start[2] + scale * y_unit_vec[2]],
        "g",
    )
    ax.plot(
        [start[0], start[0] + scale * z_unit_vec[0]],
        [start[1], start[1] + scale * z_unit_vec[1]],
        [start[2], start[2] + scale * z_unit_vec[2]],
        "b",
    )


def plotTrajectory(ax, trajectory):
    for pose in trajectory:
        plotCoordinateSystemFromTransform(pose, ax)
