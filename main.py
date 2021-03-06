import sys
import math
import warnings
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow.keras as K
from sklearn.model_selection import train_test_split

from utils import *
from autoencoder import *
from classify import Classifier
from cnn import CNNClassifier
from similar import SimilarImages

warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(42)


def main():
    X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    # num_features = [64,128,256,512,1024]
    # autoencoder = AutoencoderUpsample(256)
    #    #print("Number of layers =", "Loss =", loss, "Accuracy =", acc, "PSNR =", psnr)
    # autoencoder.train(X_train, X_test)
    # autoencoder.saveModel("autoencoder_upsample_2.h5")

    # autoencoder.loadModel("autoencoder_upsample_2.h5")
    ##encoder = autoencoder.getEncoder()

    # print("Autoencoder Trained")
    # autoencoder.showComparison(X_test[123])
    # classifier = Classifier(autoencoder)
    # classifier.train(X_train, y_train, X_test, y_test)
    # print(getCo    mpressionEfficiency(X_train, encoder))
    # X_train_enc = encoder.predict(X_train)
    # X_test_enc = encoder.predict(X_test)
    # correct = 0
    ## we are forced to use a loop here bcoz memory was running out and we were too lazy to fix it
    # for i in tqdm(range(10)):
    #    cl = closest(X_train_enc, np.array([X_test_enc[i]]))
    #    correct += (y_train[cl] == y_test[i]
    # models = ["SVM", "LR", "RF", "KNN", "DNN"]
    # for model in models:
    # clf = Classifier("DNN",autoencoder)
    # clf.saveArchitecture("DNN.png")
    # decoder = autoencoder.getDecoder()
    # print(decoder.summary())
    slr = SimilarImages()
    image = slr.getRandomImage()
    closestImages = slr.getKSimilarImages(image)
    print(closestImages)
    slr.plotResults(image, closestImages * 255)


if __name__ == "__main__":
    main()
