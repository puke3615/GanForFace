from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import keras
import sys
import os

import numpy as np

from data_loader import load_data


class DCGAN():
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        filename_weight = ''
        self.weight_path = 'weight/%s' % self.name
        self.file_weight = os.path.join(self.weight_path, filename_weight)
        self._load_weight()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(8 * 128 * 128, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((128, 128, 8)))

        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Conv2DTranspose(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # # model.add(UpSampling2D())
        # # model.add(Conv2D(128, kernel_size=3, padding="same"))
        # model.add(Conv2DTranspose(128, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))

        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Conv2DTranspose(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Conv2DTranspose(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # inception = keras.applications.Xception(input_shape=self.img_shape, weights=None, include_top=False,
        #                                         pooling='avg')
        # x = Dense(1, activation='sigmoid')(inception.output)
        # return Model(inception.input, x)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def _load_weight(self):
        if os.path.exists(self.file_weight) and os.path.isfile(self.file_weight):
            self.combined.load_weights(self.file_weight, by_name=True)
            print('Load weight [%s] successfully.' % self.file_weight)
        else:
            print('No weight loaded.')

    def _save_weight(self, epoch, d_loss):
        filename_weight = '%s_%05d-%.4f.h5' % (self.name, epoch, d_loss)
        file_weight = os.path.join(self.weight_path, filename_weight)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        self.combined.save_weights(file_weight, overwrite=True)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        generator = load_data(batch_size, self.img_rows, self.img_cols)

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            imgs, _ = next(generator)
            imgs = imgs / 127.5 - 1.
            if imgs.shape[0] != batch_size:
                continue

            # Sample noise and generate a batch of new images
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            # X = np.concatenate((imgs, gen_imgs))
            # y = np.concatenate((valid, fake))
            # d_loss = self.discriminator.train_on_batch(X, y)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # if epoch < 200:
            #     print('Train %d, d_loss: %f' % (epoch, d_loss[0]))
            # else:
            # Train the generator (wants discriminator to mistake images as real)
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(noise, valid)
            self.discriminator.trainable = True

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self._save_weight(epoch, d_loss[0])

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        save_dir = 'images/%s' % self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig("%s/%d.png" % (save_dir, epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000000, batch_size=32, save_interval=50)
