import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as K
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
import math


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.backend.log((max_pixel ** 2) / (K.backend.mean(K.backend.square(y_pred - y_true), axis=-1)))) / 2.303

def loadImages(pathX, numRows=10000):
    X = np.array(pd.read_csv(pathX, header=None, nrows=numRows))
    X = np.reshape(X, (X.shape[0], 28, 28, 4))
    X = np.divide(X, 255)
    return X



def loadData(pathX, pathY, numRows=99999, split=0.33):
    """

    :param pathX: path of data features
    :param pathY: path of data labels
    :param numRows: number of top rows to be taken (default is entire file)
    :param split: fraction of data to be used for validation purpose
    :return: tuple of train features, validation features, train labels, validation features
    """

    X = np.array(pd.read_csv(pathX, header=None, nrows=numRows))
    ## reshape to 4 channel image
    X = np.reshape(X, (X.shape[0], 28, 28, 4))
    ## normalize
    X = np.divide(X, 255)
    y = np.array(pd.read_csv(pathY, header=None, nrows=numRows))
    ## one-hot to label
    y = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    return X_train, X_test, y_train, y_test

def getCompressionEfficiency(X_train, encoder):
    """

    :param X_train: input data
    :param encoder: the encoder network which converts image to a fixed dimension vector
    :return: efficiency of the encoder that is 1 - size of encoding/size of original data
    """
    X_train_enc = encoder.predict(X_train)
    return 1-X_train_enc.nbytes/X_train.nbytes

def similarity(matrix1, matrix2):
    """

    :param matrix1: matrix of first vector
    :param matrix2: matrix of second vector
    :return: similarity between each pair of vectors
    """
    return 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')


def closest(matrix1, matrix2):
    sim = np.sum((matrix1 - matrix2)**2, axis=1)
    print(sim.shape)
    return np.argmin(sim, axis=0)


def main():
    X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    main()