import pandas as pd
import numpy as np
import tensorflow.keras as K
from utils import DataLoader
from autoencoder import AutoEncoder

def main():
    X_train = DataLoader.fileToNumpy("X_train_sat4.csv")
    autoencoder = AutoEncoder()
    autoencoder.train(X_train)




if __name__ == "__main__":
    main()