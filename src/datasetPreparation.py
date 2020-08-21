"""
Prepares datasets of modeshapes so that they can be used by NN for classification
Creates one clean dataset and four noisy versions of the same dataset, with a white noise with variance
respectively of -60, -30, -10 and -5 dB
All datasets are split into training and test sets and reshaped as needed by the classificators
"""

import datasetGenerator as dsGen
import numpy as np
import pickle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy.signal import decimate


def downsample(im, factor):
    return im[::factor, ::factor]


def downsample2(im, factor):
    im_down = decimate(im, factor, ftype='fir')
    im_down = decimate(im_down, factor, ftype='fir', axis=0)
    return im_down


def load_dataset(filepath):
    """
    Loads a dataset store in a file
    :param filepath: a string containing the file path
    :return: the dataset
    """
    with open(filepath, 'rb') as data:
        d = pickle.load(data)
    return d


def noisy_dataset(dataset, noise_db):
    """
    Given a dataset of modeshapes stored in filepath, adds a gaussian noise of variance noise_dB
    :param dataset: the clean dataset
    :param noise_db: the variance in dB of the gaussian noise to add
    :return: the noisy dataset
    """
    var = 10 ** (noise_db / 10)
    ds_new = []
    for i in range(len(dataset)):
        noise = np.random.normal(0, np.sqrt(var), dataset[i][2].shape)
        ds_new.append((dataset[i][0], dataset[i][1], dataset[i][2] + noise, dataset[i][3], dataset[i][4]))
    return ds_new


def cod_y(y):
    """
    Creates a dictionary in which each tuple of modes (m,n) is associated to a unique number
    :param y: a vector of modes
    :return: the vector of modes expressed by their scalar class, the number of classes, the dictionary
    """
    classes = np.unique(y)
    num_classes = len(classes)
    classes_dic = {}
    for i in range(num_classes):
        classes_dic[classes[i][0], classes[i][1]] = i

    y2 = []
    for i in range(len(y)):
        y2.append(classes_dic[y[i][0], y[i][1]])

    y2 = np.array(y2)
    return y2, num_classes, classes_dic


def dataset_creation(ds):
    """
    Given a dataset, splits it into training and test sets
    and reshapes them as needed by the classification models
    :param ds: the filepath where the starting dataset (can be a clean or a noisy dataset)
    :return: the train and test sets for inputs, outputs, output reshaped as matrixes, the number of classes and
    the dictionary of classes
    """
    X = []
    y = []

    for i in range(len(ds)):
        X.append(ds[i][2])
        y.append((ds[i][0][0], ds[i][0][1]))

    X = np.array(X)
    y = np.array(y, dtype='int8, int8')

    y2, num_classes, dic = cod_y(y)

    X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2)
    #X = X.reshape((X.shape[0], 1, 8, 12))
    X_train = X_train.reshape(X_train.shape[0], 1, 8, 12)
    X_test = X_test.reshape(X_test.shape[0], 1, 8, 12)

    y2_train_mat = np_utils.to_categorical(y2_train, num_classes)
    y2_test_mat = np_utils.to_categorical(y2_test, num_classes)
    #y2_mat = np_utils.to_categorical(y2, num_classes)

    return X_train, X_test, y2_train, y2_test, y2_train_mat, y2_test_mat, num_classes, dic
