"""
Functions to generate a dataset of mode shapes of plates of various dimensions
"""

import modeshapesPlate as plate
import numpy as np
import pickle


def plates_generator(fmax, lx_min, lx_max, ly_min, ly_max, step_plate, h, delta, E, rho, nu):
    """
    Creates a dataset containing all modes representable given maximum temporal frequency and sampling delta
    for all rectangular plates in range given by maximum and minimum dimensions
    :param fmax: maximum temporal frequency to reach
    :param lx_min: minimum dimension on x axis
    :param lx_max: maximum dimension on x axis
    :param ly_min: minimum dimension on y axis
    :param ly_max: maximum dimension on y axis
    :param step_plate: step in dimension of plates
    :param h: thickness of plate
    :param delta: sampling step
    :param E: Young modulus
    :param rho: density
    :param nu: Poisson modulus
    :return: the dataset, each row is a tuple containing the couple of modes, the temporal frequency, the modeshape,
    the two dimensions of plate
    """
    dataset = []
    lx = np.arange(lx_min, lx_max + 0.0001, step_plate)
    ly = np.arange(ly_min, ly_max + 0.0001, step_plate)

    for x in lx:
        for y in ly:

            modes = plate.modes_shapes(fmax, x, y, h, delta, E, rho, nu)

            for i, v in modes.items():
                t = (i, v[0], v[1], x, y)
                dataset.append(t)

    return dataset


def dataset_whole_plate(dataset):
    """
    Creates a dataset of plates of different dimensions but same number of points
    (points cover the whole area of the plate
    Dataset is also written in a file saved in folder DatasetFiles
    :param dataset: the starting dataset (whose plates are made of different number of points)
    :return: a list of tuple, each containing: couple of order of modes (m,n), temporal frequency, mode shape,
    length of plate along x and long y dimension
    """
    res = []
    num_puntix = dataset[0][2].shape[1]
    num_puntiy = dataset[0][2].shape[0]

    for row in dataset:
        x = np.linspace(0, row[3], num_puntix)
        y = np.linspace(0, row[4], num_puntiy)
        x , y = np.meshgrid(x, y)
        m, n = row[0][0], row[0][1]
        shape = plate.z(x, y, m, n, row[3], row[4], A=1)
        res.append(((m, n), row[1], shape, row[3], row[4]))

# TODO rimettere path relativo!
    #with open("./DatasetFiles/dataset_small_images", 'wb') as output:
        #pickle.dump(res, output)

    return res


def dataset_same_dim(dataset, delta):
    """
    Creates a dataset of plates of different dimensions and same number of points, sampled at the same step
    (the points don't cover the whole area of the plate and are sampled from the center)
    :param dataset: the starting dataset (whose plates are made of different number of points)
    :param delta: the sampling step between two consecutive points
    :return: a list of tuple, each containing: couple of order of modes (m,n), temporal frequency, mode shape,
    length of plate along x and long y dimension
    """
    res = []
    lx_min = dataset[0][3]
    ly_min = dataset[0][4]

    for row in dataset:
        x_min = row[3]/2 - lx_min/2
        y_min = row[4]/2 - ly_min/2
        x_max = row[3] / 2 + lx_min / 2
        y_max = row[4] / 2 + ly_min / 2
        x = np.arange(x_min, x_max, delta)
        y = np.arange(y_min, y_max, delta)
        x, y = np.meshgrid(x, y)
        m, n = row[0][0], row[0][1]
        shape = plate.z(x, y, m, n, row[3], row[4], A=1)
        res.append(((m, n), row[1], shape, row[3], row[4]))

    return res


nu = 0.3
E = 2.1e11
rho = 7850
h = 0.002
delta = 0.02
dataset = plates_generator(3500, 0.23, 0.36, 0.15, 0.20, 0.01, h, delta, E, rho, nu)
# 1927 rows
