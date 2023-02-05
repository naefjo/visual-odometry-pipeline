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

    img0, img1, K, last_frame, ground_truth, bootstrap_frames = load_dataset(dataset)

    # Bootstrap
    sift_params = {
        "nfeatures": 0,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.02,  # default 0.04
        "edgeThreshold": 50,  #
        "sigma": 1.6,
    }
    feature_detector = SIFTKeypointDetectorAndMatcher(sift_params)

    # keypoints_prev, landmarks_prev, T_c0_c1, inliner_stats = bootstrapVoPipeline(
    #     img0, img1, K, feature_detector
    # )

    previous_image_features = bootstrapVoPipeline(img0, img1, K, feature_detector)

    # Set world coordinate frame arbitrarily for visualization

    trajectory = [generateWorldFrame(), previous_image_features.transform]
    transform_C0_C1 = (invertSE3Matrix(trajectory[0]) @ trajectory[1])[:3, -1]
    max_distance_threshold_new_frames = np.linalg.norm(transform_C0_C1)

    fig, axs = generateFigure()
    add_landmarks_to_plot(axs, previous_image_features.landmarks)
    plotTrajectory(axs, trajectory, previous_image_features)
    plt.waitforbuttonpress(-1)

    # Save keyframe information
    keyframe_image_features = previous_image_features
    keyframe_image = img1

    # Continuous operation
    prev_img = img1
    for image_index in range(bootstrap_frames[1] + 1, last_frame):
        print("\n\nProcessing frame {}\n=====================\n".format(image_index))
        image = getContinuousOperationImage(image_index, dataset)

        image_features = processFrame(
            prev_img,
            image,
            K,
            previous_image_features,
            feature_detector,
        )

        transform_past_trajectory_to_current_frame = (
            invertSE3Matrix(trajectory[-1]) @ image_features.transform
        )

        if (
            np.linalg.norm(transform_past_trajectory_to_current_frame[:3, -1])
            < max_distance_threshold_new_frames
        ):
            trajectory.append(image_features.transform)
            # Makes sure that plots refresh.
            plotCoordinateSystemFromTransform(trajectory[-1], axs[0])
            plt.pause(0.01)

        else:
            print("consecutive frames are too far apart. Skipping trajectory update.")
            image_features = previous_image_features
            image = prev_img

        if image_features.landmarks.shape[1] < kLandmarkThreshold:
            print("localizing new landmarks")
            image_features = localizeNewLandmarks(
                K,
                keyframe_image,
                keyframe_image_features,
                image,
                image_features,
                feature_detector,
            )

            transform_keyframe_to_current_frame = (
                invertSE3Matrix(keyframe_image_features.transform)
                @ image_features.transform
            )
            if np.linalg.norm(transform_keyframe_to_current_frame[:3, -1]) > 1:
                keyframe_image_features = image_features
                keyframe_image = image

            add_landmarks_to_plot(axs, image_features.landmarks)

        prev_img = image
        previous_image_features = image_features


if __name__ == "__main__":
    main()
