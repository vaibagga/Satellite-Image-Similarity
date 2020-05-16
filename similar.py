import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as K
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.spatial as sp
import matplotlib.pyplot as plt
import math
from utils import *
from autoencoder import AutoencoderUpsample
#import cv2

class SimilarImages:
    def __init__(self, autoencoderPath="autoencoder_upsample_2.h5"):
        autoencoder = AutoencoderUpsample(256)
        autoencoder.loadModel(autoencoderPath)
        self.encoder = autoencoder.getEncoder()
        self.decoder = autoencoder.getDecoder()

    def getRandomImage(self, path="X_test_sat4.csv"):
        images = loadImages(path)
        idx = np.random.randint(images.shape[0], size=1)
        return images[idx,:]

    def saveImageEncodings(self, imageFilePath="X_train_sat4.csv", savePath="X_train_encoding.npy"):
        X = loadImages(imageFilePath)
        print("Data Loaded")
        encoding = self.encoder.predict(X)
        np.save(savePath, encoding)

    def loadImageEncodings(self, savePath="X_train_encoding.npy"):
        encodings = np.load(savePath)
        return encodings

    def largest_indices(self, ary, n):
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def getKSimilarImages(self, image, k=5, encodingPath="X_train_encoding.npy"):
        encodings = self.loadImageEncodings(encodingPath)
        encodingsNorm = normalize(encodings, axis=1, norm='l2')
        image = np.divide(image, 255)
        imageEncoding = self.encoder.predict([image])
        imageEncoding = normalize(imageEncoding, axis=1, norm='l2')
        print(imageEncoding.shape, encodingsNorm.T.shape)
        simMatrix = np.matmul(imageEncoding, encodingsNorm.T).ravel()
        ind = np.argpartition(simMatrix, -k)[-k:]
        closestImageEncoding = encodings[ind,:]
        closestImages = self.decoder.predict(closestImageEncoding)
        return closestImages

    def plotResults(self, image, similarImages):
        rows = 2
        numImages = similarImages.shape[0]
        plt.subplot(rows, 6, 1)
        plt.axis('off')
        plt.imshow(image[0,:,:,:3])
        for idx in range(numImages):
            plt.subplot(rows, 6, idx + 2)
            plt.axis('off')
            plt.imshow(similarImages[idx,:,:,:3])
        plt.show()