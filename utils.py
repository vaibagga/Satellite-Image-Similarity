import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.X = None

    def fileToNumpy(self, path):
        self.X = np.array(pd.read_csv(path, header=None)
        self.X = np.reshape(self.X, (self.X.shape[0], 28, 28, 4))
        ##self.X = np.float(self.X)
        self.X = np.divide(self.X, 255)
        return self.X

    def saveToFile(self, path):
        np.save(path, self.X)


def main():
    dl = DataLoader()
    dl.fileToNumpy("X_train_sat4.csv")
    dl.saveToFile("X_train.npy")


if __name__ == "__main__":
    main()