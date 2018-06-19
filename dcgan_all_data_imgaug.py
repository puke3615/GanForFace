from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import *
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


class DCGAN_Expand():
    def __init__(self):
        self.name = os.path.basename(__file__).split('.')[0].lower()
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
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5), metrics=['accuracy'])

        filename_weight = ''
        self.weight_path = 'weight/%s' % self.name
        self.file_weight = os.path.join(self.weight_path, filename_weight)
        self._load_weight()

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))

        # convs = []
        # n_conv = 1
        # n_item = self.latent_dim // n_conv
        # # n_conv_filters = 100 // n_conv
        # for i in range(n_conv):
        #     start = i * n_item
        #     end = (i + 1) * n_item
        #     x_item = Lambda(lambda x: x[:, start: end], name='%d_%d' % (start, end))(noise)
        #     x_item = Reshape((32, 32, 1))(x_item)
        #     x_item = Conv2DTranspose(filters=2048, kernel_size=3, strides=1, padding='same')(x_item)
        #     x_item = BatchNormalization(momentum=0.8)(x_item)
        #     x_item = Dropout(0.25)(x_item)
        #     convs.append(x_item)
        #
        # x = convs[0] if len(convs) == 1 else Concatenate()(convs)

        x = Dense(8 * 128 * 128, activation='relu')(noise)

        x = Reshape((128, 128, 8))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(128, kernel_size=5, strides=1, padding="same", activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(64, kernel_size=5, strides=1, padding="same", activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(32, kernel_size=5, padding="same", activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(16, kernel_size=5, padding="same", activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(8, kernel_size=5, padding="same", activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2DTranspose(self.channels, kernel_size=5, padding="same")(x)
        img = Activation("tanh")(x)

        model = Model(noise, img)
        model.summary()
        return model

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

        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def _load_weight(self):
        if os.path.exists(self.file_weight) and os.path.isfile(self.file_weight):
            self.combined.load_weights(self.file_weight, by_name=True)
            print('Load weight [%s] successfully.' % self.file_weight)
        else:
            print('No weight loaded.')

    def _save_weight(self, epoch):
        filename_weight = '%s_%05d.h5' % (self.name, epoch)
        file_weight = os.path.join(self.weight_path, filename_weight)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        self.combined.save_weights(file_weight, overwrite=True)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        generator = load_data(batch_size, self.img_rows, self.img_cols, imgaug=True)

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
            # noise = np.random.normal(0, 0.02, (batch_size, self.latent_dim))
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
            print("%07d [D loss: %7.4f, acc: %6.2f%%] [G loss: %7.4f, acc: %6.2f%%]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], 100 * g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self._save_weight(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
        # noise = np.random.normal(0, 0.02, (r * c, self.latent_dim))
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
    dcgan = DCGAN_Expand()
    dcgan.train(epochs=4000000, batch_size=32, save_interval=50)
