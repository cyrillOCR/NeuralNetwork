#!/usr/bin/env python3
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

N_LABELED = 1000
N_UNLABELED = 9000
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train_labeled, x_train_unlabeled = (x_train[:N_LABELED], x_train[N_LABELED:N_LABELED + N_UNLABELED])
x_test_labeled, x_test_unlabeled = (x_test[:N_LABELED], x_test[N_LABELED:N_LABELED + N_UNLABELED])
y_train_labeled = to_categorical(y_train[:N_LABELED])
y_test_labeled = to_categorical(y_test[:N_LABELED])

noise_factor = 0.3
x_train_noisy = x_train_unlabeled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_unlabeled.shape)
x_test_noisy = x_test_unlabeled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_unlabeled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


input_img = Input(shape=(784,))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)

x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(784, activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
autoencoder.fit(x_train_noisy, x_train_unlabeled, epochs=10, batch_size=32, validation_data=(x_test_noisy, x_test_unlabeled))

y = Dense(10, activation='softmax')(encoded)
supervised = Model(input_img, y)
supervised.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
supervised.fit(x_train_labeled, y_train_labeled, epochs=10, batch_size=32, validation_data=(x_test, to_categorical(y_test)))
test_loss, test_accuracy = supervised.evaluate(x_test, to_categorical(y_test))
print('supervised + DAE')
print('loss value: {}\naccuracy: {}\n'.format(test_loss, test_accuracy))
input('Press enter to continue...')

z = Dense(128, activation='relu')(input_img)
z = Dense(64, activation='relu')(z)
z = Dense(32, activation='relu')(z)
z = Dense(10, activation='softmax')(z)
supervised2 = Model(input_img, z)
supervised2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
supervised2.fit(x_train_labeled, y_train_labeled, epochs=10, batch_size=32, validation_data=(x_test, to_categorical(y_test)))
test_loss, test_accuracy = supervised2.evaluate(x_test, to_categorical(y_test))
print('supervised only')
print('loss value: {}\naccuracy: {}\n'.format(test_loss, test_accuracy))
input('Press enter to continue...')
