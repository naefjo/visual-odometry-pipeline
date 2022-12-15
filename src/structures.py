from enum import Enum


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
