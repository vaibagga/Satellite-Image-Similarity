from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras as K
from autoencoder import *
from utils import loadData

class Classifier():
    """
    Dense Neural Net Classifier using encoding of autoencoder
    """
    def __init__(self, model_name, autoencoder):
        """

        :param autoencoder: encoder to be used by classifier
        """
        self.modelName = model_name
        self.model = None
        if model_name == "SVM":
            self.model = SVC()
        if model_name == "LR":
            self.model = LogisticRegression()
        if model_name == "RF":
            self.model = RandomForestClassifier()
        if model_name == "KNN":
            self.model = KNeighborsClassifier()
        if model_name == "DNN":
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
        """

        :param X_train: training features
        :param y_train: training labels
        :param X_test: validation features
        :param y_test: validation labels
        :return:
        """

        X_train, X_test = self.encoder.predict(X_train), self.encoder.predict(X_test)
        print("Training", self.modelName)
        if self.modelName == "DNN":
            self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        else:
            self.model.fit(X_train, y_train)
        if self.modelName == "DNN":
            print("Train Accuracy of DNN =", self.model.evaluate(X_train, y_train)[1])
            print("Validation Accuracy of DNN =", self.model.evaluate(X_test, y_test)[1])
        else:
            print("Train Accuracy of", self.modelName, "=", self.model.score(X_train, y_train))
            print("Validation Accuracy", self.modelName, "=", self.model.score(X_test, y_test))



    def predict(self, X_test):
        """

        :param X_test: prediction data
        :return: predicted labels
        """
        X_test = self.encoder.predict(X_test)
        return self.model.predict(X_test)




