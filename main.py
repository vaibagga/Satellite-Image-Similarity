import warnings
import random

import pandas as pd
import numpy as np
import tensorflow.keras as K

from utils import loadData
from autoencoder import *
from classify import Classifier
from cnn import CNNClassifier

warnings.filterwarnings("ignore")
random.seed(42)


def main():
    X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    print("Data Loaded")

    autoencoder = AutoencoderUpsample()
    autoencoder.train(X_train, X_test)
    autoencoder.saveModel("autoencoder_upsample_2.h5")
    print("Autoencoder Trained")

    #classifier = Classifier(autoencoder)
    #classifier.train(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()