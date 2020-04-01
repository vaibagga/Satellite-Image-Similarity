import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoder:
    """
    Base class for autoencoder
    """
    def __init__(self):
        ## autoencoder model
        self.autoencoder = K.models.Sequential()

        ## encoder site
        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,4)))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
        self.autoencoder.add(K.layers.Flatten())

        self.autoencoder.add(K.layers.Dense(49, activation='softmax'))

        # decoder site
        self.autoencoder.add(K.layers.Reshape((7, 7, 1)))
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same'))

    def getMiddleLayer(self):
        """

        :return: hidden representation layer of the autoencoder
        """
        for layer in self.autoencoder.layers:
            if isinstance(layer, K.layers.Dense):
                return layer


    def getEncoder(self):
        """

        :return: the encoder network of the autoencoder
        """
        middleLayer = self.getMiddleLayer()
        encoder = K.models.Model(inputs=self.autoencoder.input, outputs=middleLayer.output)
        #decoder = K.models.Model(inputs=middleLayer.output, outputs=self.autoencoder.output)
        return encoder

    def train(self, X_train, X_test):
        """
        trains the autoencoder model
        :param X_train: training data
        :param X_test: validation data
        :return:
        """
        es = K.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        history = self.autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=50, batch_size=64, verbose=1, callbacks=[es])

    def saveModel(self, path):
        """
        Saves model to path
        :param path: path of file to save the model
        :return:
        """
        self.autoencoder.save(path)

    def loadModel(self, path):
        """

        :param path:
        :return:
        """
        self.autoencoder = K.models.load_model(path)

    def showComparison(self, image):
        """
        shows comparison of original image and reconstructed image
        :param image: 28X28 4-channel image
        :return:
        """
        image = np.reshape(image, (1, 28, 28, 4))
        pred = self.autoencoder.predict(image)
        print(image.shape)
        fig, axs = plt.subplots(2,2)
        #axs[0].title("Input Image")
        #axs[0].imshow(image[0]*255)
        #axs[1].title("Predicted Image")
        axs[0][0].imshow(image[0, :, :, :3])
        axs[0][0].set_title("Original RGB Image")
        axs[0][1].imshow(pred[0, :, :, :3])
        axs[0][1].set_title("Reconstructed RGB Image")
        axs[1][0].imshow(image[0,:,:,3], cmap='gray')
        axs[1][0].set_title("Original near IR Image")
        axs[1][1].imshow(pred[0,:,:,3], cmap='gray')
        axs[1][1].set_title("Reconstructed near IR Image")
        plt.show()

    def saveArchitecture(self, path):
        """
        Saves an image of the model architecture
        :return:
        """
        print(self.autoencoder.summary())
        K.utils.plot_model(self.autoencoder, to_file=path, show_shapes=True)


class AutoencoderCNN(AutoEncoder):
    """
    CNN Based Autoencoder
    """
    def __init__(self):
        AutoEncoder.__init__(self)
        self.autoencoder = K.models.Sequential()

        ## encoder site
        self.autoencoder.add(K.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 4)))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2D(32, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Dropout(0.4))

        self.autoencoder.add(K.layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Dropout(0.4))

        self.autoencoder.add(K.layers.Conv2D(64, kernel_size=4, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Flatten())
        self.autoencoder.add(K.layers.Dropout(0.4))
        self.autoencoder.add(K.layers.Reshape((8, 8, 1)))


        ## decoder site
        self.autoencoder.add(K.layers.Conv2DTranspose(64, kernel_size=5, strides=2, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(64, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(64, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())

        self.autoencoder.add(K.layers.Conv2DTranspose(32, kernel_size=5, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(32, kernel_size=3, activation='relu'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(32, kernel_size=3, activation='relu'))

        self.autoencoder.add(K.layers.Conv2D(4, kernel_size=4, activation='sigmoid'))

class AutoencoderUpsample(AutoEncoder):
    """
    Autoencoder using Upsampling
    """
    def __init__(self, num_features):
        AutoEncoder.__init__(self)

        ## encoder site
        self.autoencoder = K.models.Sequential()
        self.autoencoder.add(K.layers.Conv2D(8, (3,3), padding='same', activation='relu', input_shape=(28,28,4)))
        self.autoencoder.add(K.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2,2)))
        self.autoencoder.add(K.layers.Conv2D(16, (3,3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Flatten())
        self.autoencoder.add(K.layers.Dense(num_features))

        ## decoder site
        self.autoencoder.add(K.layers.Dense(784))
        self.autoencoder.add(K.layers.Reshape((7, 7, 16)))
        self.autoencoder.add(K.layers.UpSampling2D((2,2)))
        self.autoencoder.add(K.layers.Conv2D(16, (3,3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.UpSampling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.Conv2D(4, (3, 3), padding='same', activation='sigmoid'))



def main():
    autoencoder = AutoEncoder()

if __name__ == "__main__":
    main()