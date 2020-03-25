import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoder:
    def __init__(self):
        ## autoencoder model
        self.autoencoder = K.models.Sequential()

        ## encoder

        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,4)))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
        self.autoencoder.add(K.layers.Flatten())
        #self.autoencoder.add(K.layers.Dense(49, activation='relu'))
        self.autoencoder.add(K.layers.Dense(49, activation='softmax'))
        #self.autoencoder.add(K.layers.Dense(49, activation='relu'))
        # DECODER
        self.autoencoder.add(K.layers.Reshape((7, 7, 1)))
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same'))

    def train(self, X_train, X_test):
        es = K.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        history = self.autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, verbose=1, callbacks=[es])

    def saveModel(self, path):
        self.autoencoder.save(path)

    def loadModel(self, path):
        self.autoencoder = K.models.load_model(path)

    def showComparison(self, image):
        image = np.reshape(image, (1, 28, 28, 4))
        pred = self.autoencoder.predict(image)
        image = image[:, :, :, :3]
        pred = pred[:, :, :, :3]
        print(image.shape)
        fig, axs = plt.subplots(2)
        #axs[0].title("Input Image")
        #axs[0].imshow(image[0]*255)
        #axs[1].title("Predicted Image")
        axs[0].imshow(pred[0])
        axs[1].imshow(image[0])
        plt.show()

    def saveArchitecture(self):
        print(self.autoencoder.summary())


def main():
    autoencoder = AutoEncoder()

if __name__ == "__main__":
    main()