import numpy as np
import cv2
import matplotlib.pyplot as plt

from structures import *
from utils import *
from vo_bootstrap import bootstrap_vo_pipeline


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
    img0 = cv2.imread(get_image_path(0, dataset), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(get_image_path(5, dataset), cv2.IMREAD_GRAYSCALE)

else:
    raise NotImplementedError

# cv2.imshow('', img0)
# cv2.waitKey(0)
# cv2.imshow('', img1)
# cv2.waitKey(0)

bootstrap_vo_pipeline(img0, img1, K)
# # Continuous operation
# range = (bootstrap_frames(2)+1):last_frame;
# for i = range
#     fprintf('\n\nProcessing frame %d\n=====================\n', i);
#     if ds == 0
#         image = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);
#     elseif ds == 1
#         image = rgb2gray(imread([malaga_path ...
#             '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
#             left_images(i).name]));
#     elseif ds == 2
#         image = im2uint8(rgb2gray(imread([parking_path ...
#             sprintf('/images/img_%05d.png',i)])));
#     else
#         assert(false);
#     end
#     % Makes sure that plots refresh.
#     pause(0.01);

#     prev_img = image;
# end
