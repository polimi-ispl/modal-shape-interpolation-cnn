"""
Functions for the preparation of mode shape images (i.e. removal of zero edges)
"""


def zero_edges(dataset):
    """
    Sets edges of all images in a given dataset to zero
    :param dataset: the dataset (usually made of interpolated images)
    :return: the dataset with all edges set to zero
    """
    for i in range(dataset.shape[0]):
        dataset[i][0, :, 0] = 0
        dataset[i][-1, :, 0] = 0
        dataset[i][:, 0, 0] = 0
        dataset[i][:, -1, 0] = 0

    return dataset


def cut_edges_output(im):
    """
    Cuts edges of the image, which are zero
    :param im: input image
    :return: the image with no edges
    """
    return im[1:-1, 1:-1]

