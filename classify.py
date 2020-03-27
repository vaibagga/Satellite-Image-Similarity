from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow.keras as K
from autoencoder import *
from utils import loadData

class Classifier():
    def __init__(self, autoencoder):
        self.model = K.models.Sequential([
                        K.layers.Dense(128, input_shape=(256,)),
                        K.layers.Activation('relu'),
                        K.layers.Dense(64),
                        K.layers.Activation('relu'),
                        K.layers.Dense(32),
                        K.layers.Activation('relu'),
                        K.layers.Dense(4),
                        K.layers.Activation('softmax')])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.encoder = autoencoder.getEncoder()

    def train(self, X_train, y_train, X_test, y_test):
        X_train, X_test = self.encoder.predict(X_train), self.encoder.predict(X_test)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        print("Train Accuracy =", self.model.evaluate(X_train, y_train)[1])
        print("Validation Accuracy =", self.model.evaluate(X_test, y_test)[1])


    def predict(self, X_test):
        X_test = self.encoder.predict(X_test)
        return self.svmModel.predict(X_test)
