from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os

import numpy as np

from data_loader import load_data


class GAN():
    def __init__(self):
        self.name = self.__class__.__name__.lower()
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
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        filename_weight = ''
        self.weight_path = 'weight/%s' % self.name
        self.file_weight = os.path.join(self.weight_path, filename_weight)
        self._load_weight()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # import keras
        # inception = keras.applications.Xception(input_shape=self.img_shape, weights=None, include_top=False,
        #                                         pooling='avg')
        # x = Dense(1, activation='sigmoid')(inception.output)
        # return Model(inception.input, x)

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
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

    def _save_weight(self, epoch, g_loss):
        filename_weight = '%s_%05d-%.4f.h5' % (self.name, epoch, g_loss)
        file_weight = os.path.join(self.weight_path, filename_weight)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        self.combined.save_weights(file_weight, overwrite=True)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
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

            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]

            imgs, _ = next(generator)
            # from PIL import Image
            # image = Image.fromarray(imgs[0].astype(np.uint8))
            # image.save('images/%d.png' % (epoch))

            imgs = imgs / 127.5 - 1.
            if imgs.shape[0] != batch_size:
                continue

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self._save_weight(epoch, g_loss)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        save_dir = 'images/%s' % self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig("%s/%d.png" % (save_dir, epoch))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=3000000, batch_size=32, sample_interval=50)
