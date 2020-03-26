import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoder:
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
        #self.autoencoder.add(K.layers.Dense(49, activation='relu'))

        self.autoencoder.add(K.layers.Dense(49, activation='softmax'))

        #self.autoencoder.add(K.layers.Dense(49, activation='relu'))

        # decoder site
        self.autoencoder.add(K.layers.Reshape((7, 7, 1)))
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
        self.autoencoder.add(K.layers.BatchNormalization())
        self.autoencoder.add(K.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same'))

    def getMiddleLayer(self):
        for layer in self.autoencoder.layers:
            if isinstance(layer, K.layers.Dense):
                return layer


    def getEncoder(self):
        middleLayer = self.getMiddleLayer()
        encoder = K.models.Model(inputs=self.autoencoder.input, outputs=middleLayer.output)
        #decoder = K.models.Model(inputs=middleLayer.output, outputs=self.autoencoder.output)
        return encoder

    def train(self, X_train, X_test):
        es = K.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        annealer = K.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        history = self.autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size=64, verbose=1, callbacks=[es, annealer])

    def saveModel(self, path):
        self.autoencoder.save(path)

    def loadModel(self, path):
        self.autoencoder = K.models.load_model(path)

    def showComparison(self, image):
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

    def saveArchitecture(self):
        print(self.autoencoder.summary())
        K.utils.plot_model(self.autoencoder, to_file='model.png', show_shapes=True)


class AutoencoderCNN(AutoEncoder):
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
    def __init__(self):
        AutoEncoder.__init__(self)
        self.autoencoder = K.models.Sequential()
        self.autoencoder.add(K.layers.Conv2D(8, (3,3), padding='same', activation='relu', input_shape=(28,28,4)))
        self.autoencoder.add(K.layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2,2)))
        self.autoencoder.add(K.layers.Conv2D(16, (3,3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2)))
        self.autoencoder.add(K.layers.Flatten())
        self.autoencoder.add(K.layers.Dense(256))
        ## self.autoencoder.add(K.layers.Dropout(0.3))
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