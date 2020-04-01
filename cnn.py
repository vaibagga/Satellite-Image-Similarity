import tensorflow.keras as K
import matplotlib.pyplot as plt
from utils import loadData

class CNNClassifier():
    """
    CNN for classifying image
    """
    def __init__(self):
        self.model = K.models.Sequential()
        self.model.add(K.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 4)))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Dropout(0.4))

        self.model.add(K.layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Dropout(0.4))

        self.model.add(K.layers.Conv2D(128, kernel_size=4, activation='relu'))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dropout(0.4))
        self.model.add(K.layers.Dense(4, activation='softmax'))

    def train(self, X_train, y_train, X_val, y_val, plot = True):
        """

        :param X_train: training features
        :param y_train: training labels
        :param X_val: validation features
        :param y_val: validation labels
        :param plot: plot loss curves
        :return: None
        """
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        epochs = 50
        es = K.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = K.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
        annealer = K.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        history = self.model.fit(X_train, y_train, batch_size=64, epochs=epochs, callbacks=[annealer, es, mc], verbose=0,
                            validation_data=(X_val, y_val))
        print(self.model.evaluate(X_test, y_test)[0])
        ## accuracy plot
        if plot:
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
            # Plot training & validation loss values
            plt.plot(history.history['loss  '])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

    def saveModel(self, path):
        """

        :param path: path to save the model
        :return:
        """
        self.model.save(path)

def main():
    cnn = CNNClassifier()
    X_train, X_test, y_train, y_test = loadData("X_train_sat4.csv", "y_train_sat4.csv")
    print("Data Loaded")
    cnn.train(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()

