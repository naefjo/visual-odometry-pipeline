from structures import data_path, DataSet

def get_image_path(image_index: int, dataset: DataSet) -> str:
    """
    Returns the image path for a given image index and dataset.
    """
    return data_path[dataset] + "images/img_" + str(image_index).zfill(5)+".png"