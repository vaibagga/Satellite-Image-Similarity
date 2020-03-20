import tensorflow.keras as K

class AutoEncoder:
    def __init__(self):
        ## autoencoder model
        self.autoencoder = K.models.Sequential()

        ## encoder
        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(56, 56, 4)))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2), padding='same'))
        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.MaxPooling2D((2, 2), padding='same'))

        ## decoder
        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.UpSampling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.autoencoder.add(K.layers.UpSampling2D((2, 2)))
        self.autoencoder.add(K.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, X_train):
        self.autoencoder.fit(X_train, X_train, epochs=100, verbose=1)

    def saveModel(self, path):
        self.autoencoder.save(path)


def main():
    autoencoder = AutoEncoder()

if __name__ == "__main__":
    main()