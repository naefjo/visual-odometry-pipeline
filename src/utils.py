import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    cv2.destroyWindow("feature_matches")
    cv2.imshow(
        "feature_matches",
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
    cv2.waitKey(delay=1000)


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


def plotTrajectory(axs, trajectory, previous_image_features):
    for pose in trajectory:
        plotCoordinateSystemFromTransform(pose, axs[0])


def load_dataset(dataset):
    if dataset == DataSet.KITTI:
        # TODO: convert matlab code to python code
        # # need to set kitti_path to folder containing "05" and "poses"
        # # assert(exist('kitti_path', 'var') ~= 0);
        # ground_truth = load([kitti_path '/poses/05.txt']);
        # ground_truth = ground_truth(:, [end-8 end]);
        # last_frame = 4540;
        # K = np.array(
        #     [
        #         [7.188560000000e+02 0 6.071928000000e+02],
        #         [0 7.188560000000e+02 1.852157000000e+02],
        #         [0 0 1],
        #     ]
        # )
        pass

    elif dataset == DataSet.MALAGA:
        # TODO: convert matlab code to python code
        # # Path containing the many files of Malaga 7.
        # assert(exist('malaga_path', 'var') ~= 0);
        # images = dir([malaga_path ...
        #     '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
        # left_images = images(3:2:end);
        # last_frame = length(left_images);
        # K = [621.18428 0 404.0076
        #     0 621.18428 309.05989
        #     0 0 1];
        pass

    elif dataset == DataSet.PARKING:
        # Path containing images, depths and all...
        # assert(exist('parking_path', 'var') ~= 0);
        last_frame = 598
        bootstrap_frames = [0, 5]
        # NOTE: If the following command fails, strip the trailing commas from each line
        K = np.loadtxt(data_path[dataset] + "K.txt", delimiter=",")

        ground_truth = np.loadtxt(data_path[dataset] + "poses.txt")
        ground_truth = ground_truth[:, -9:]

    else:
        raise NotImplementedError

    # need to set bootstrap_frames
    if dataset == DataSet.KITTI:
        # TODO: convert matlab code to python code
        # img0 = imread([kitti_path '/05/image_0/' ...
        #     sprintf('%06d.png',bootstrap_frames(1))]);
        # img1 = imread([kitti_path '/05/image_0/' ...
        #     sprintf('%06d.png',bootstrap_frames(2))]);
        pass

    elif dataset == DataSet.MALAGA:
        # TODO: convert matlab code to python code
        # img0 = rgb2gray(imread([malaga_path ...
        #     '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        #     left_images(bootstrap_frames(1)).name]));
        # img1 = rgb2gray(imread([malaga_path ...
        #     '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        #     left_images(bootstrap_frames(2)).name]));
        pass

    elif dataset == DataSet.PARKING:
        img0 = cv2.imread(
            getImagePath(bootstrap_frames[0], dataset), cv2.IMREAD_GRAYSCALE
        )
        img1 = cv2.imread(
            getImagePath(bootstrap_frames[1], dataset), cv2.IMREAD_GRAYSCALE
        )

    else:
        raise NotImplementedError

    # cv2.imshow('', img0)
    # cv2.waitKey(0)
    # cv2.imshow('', img1)
    # cv2.waitKey(0)

    return img0, img1, K, last_frame, ground_truth, bootstrap_frames


def generateFigure():
    fig = plt.figure()
    ax3d = fig.add_subplot(projection="3d")
    ax2d = None
    # ax2d = fig.add_subplot(2, 1, 2)

    # ax2d.set_xlabel("frame")
    # ax2d.set_ylabel("# landmarks")

    ax3d.set_xlim(0, 30)
    ax3d.set_ylim(0, 30)
    ax3d.set_zlim(-15, 15)
    ax3d.set_xlabel("inertial x")
    ax3d.set_ylabel("inertial y")
    ax3d.set_zlabel("inertial z")
    return fig, (ax3d, ax2d)


def add_landmarks_to_plot(axs, landmarks):
    axs[0].scatter3D(
        landmarks[0, :],
        landmarks[1, :],
        landmarks[2, :],
        c="black",
        s=0.5,
    )
