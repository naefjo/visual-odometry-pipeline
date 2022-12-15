import numpy as np
import cv2
import matplotlib.pyplot as plt

from structures import *
from utils import *
from vo_bootstrap import bootstrapVoPipeline
from vo_continuous_operation import processFrame, localizeNewLandmarks
from feature_detector import SIFTKeypointDetectorAndMatcher


def main():
    dataset = DataSet.PARKING

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

    # Bootstrap
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
    sift_params = {}
    feature_detector = SIFTKeypointDetectorAndMatcher(sift_params)

    keypoints_prev, landmarks_prev, T_c0_c1, inliner_stats = bootstrapVoPipeline(
        img0, img1, K, feature_detector
    )

    # Set world coordinate frame arbitrarily for visualization
    T_I_c0 = generateWorldFrame()
    landmarks_prev = T_I_c0 @ landmarks_prev

    trajectory = [T_I_c0, T_I_c0 @ T_c0_c1]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter3D(landmarks_prev[0, :], landmarks_prev[1, :], landmarks_prev[2, :])

    plotTrajectory(ax, trajectory)

    ax.set_xlim(-10, 20)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 20)
    ax.set_xlabel("inertial x")
    ax.set_ylabel("inertial y")
    ax.set_zlabel("inertial z")
    plt.waitforbuttonpress(-1)

    # Continuous operation
    prev_img = img1
    for image_index in range(bootstrap_frames[1] + 1, last_frame):
        print("\n\nProcessing frame {}\n=====================\n".format(image_index))
        image = getContinuousOperationImage(image_index, dataset)

        keypoints_prev, landmarks_prev, T_I_new, inliner_stats = processFrame(
            prev_img,
            image,
            K,
            keypoints_prev[0],
            keypoints_prev[1],
            landmarks_prev,
            trajectory[-1],
            feature_detector,
        )

        trajectory.append(T_I_new)

        plotTrajectory(ax, trajectory)

        # Makes sure that plots refresh.
        plt.pause(0.01)

        prev_img = image

        if landmarks_prev.shape[1] < kLandmarkThreshold:
            # TODO: implement the following func
            keypoints_prev, landmarks_prev = localizeNewLandmarks()


def generateWorldFrame() -> NDArray:
    T_I_c0 = np.eye(4)
    T_I_c0[:3, -1:] = np.array([0, 0, 1]).reshape(3, 1)
    T_I_c0 = T_I_c0 @ RxToHomogeneousTransform(-np.pi / 2)
    return T_I_c0


if __name__ == "__main__":
    main()
