import pandas as pd
import numpy as np
import tensorflow.keras as K
from utils import loadData
from autoencoder import *
from cnn import CNNClassifier
import numpy as np

def main():
    #X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    print("Data Loaded")
    #model = CNNClassifier()
    #model.train(X_train, y_train, X_test, y_test)
    #model.saveModel("initial.h5")
    autoencoder = AutoencoderCNN()
    #autoencoder.train(X_train, X_test)
    #autoencoder.showComparison(X_test[69])
    #autoencoder.saveModel("model_3_KL.h5")
    #autoencoder.loadModel("model2.h5")
    #autoencoder.showComparison(X_test[420])
    #autoencoder.saveArchitecture()
    #au = AutoencoderCNN()
    autoencoder.saveArchitecture()
    #autoencoder.train(X_train, X_test)
    #$autoencoder.saveModel("model_upsample.h5")
    #autoencoder.loadModel("model_CNN.h5")
    #autoencoder.showComparison(X_test[420])



if __name__ == "__main__":
    main()