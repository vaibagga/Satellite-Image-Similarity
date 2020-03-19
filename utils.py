import pandas as pd
import numpy as np
import h5py

class DataLoader:
    def __init__(self):
        self.X = None

    def fileToNumpy(self, path):
        self.X = np.array(pd.read_csv(path, header=None))
        self.X = np.reshape(X_train, (X_train.shape[0], 56, 56))

    def saveToFile(self, path, name):
        file = h5py.File(path, 'w')
        file.create_dataset(name, X)
        file.close()


def main():
    dl = DataLoader()
    dl.fileToNumpy("X_train_sat4.csv")
    dl.saveToFile("X_train.h5", "X_train")


if __name__ == "__main__":
    main()