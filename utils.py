import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def loadData(pathX, pathY, numRows=80000, split=0.33):
    X = np.array(pd.read_csv(pathX, header=None, nrows=numRows))
    X = np.reshape(X, (X.shape[0], 28, 28, 4))
    X = np.divide(X, 255)
    y = pd.read_csv(pathY, header=None, nrows=numRows)
    y = np.array(y)
    y = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    main()