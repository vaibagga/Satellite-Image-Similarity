import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.X = None

    def fileToNumpy(self, path):
        self.X = np.array(pd.read_csv(path, header=None))
        self.X = np.reshape(X_train, (X_train.shape[0], 56, 56))
        self.X /= 255

    def saveToFile(self, path):
        np.save(path, self.X)


def main():
    dl = DataLoader()
    dl.fileToNumpy("X_train_sat4.csv")
    dl.saveToFile("X_train.npy")


if __name__ == "__main__":
    main()