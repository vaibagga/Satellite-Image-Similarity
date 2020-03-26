from sklearn.svm import SVC
from autoencoder import *
from utils import loadData

class Classifier():
    def __init__(self, autoencoder):
        self.svmModel = SVC()
        self.encoder = autoencoder.getEncoder()

    def train(self, X_train, y_train, X_test, y_test):
        X_train, X_test = self.encoder.predict(X_train), self.encoder.predict(X_test)
        self.svmModel.fit(X_train, y_train)
        print("Test Accuracy =", self.svmModel.score(X_train, y_train))
        print("Validation Accuracy =", self.svmModel.score(X_test, y_test))


    def predict(self, X_test):
        X_test = self.encoder.predict(X_test)
        return self.svmModel.predict(X_test)
