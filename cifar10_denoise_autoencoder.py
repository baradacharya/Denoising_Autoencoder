# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:20:05 2017

@author: 7
"""
from __future__ import print_function
import numpy as np
#setting a seed for the computer's pseudorandom number generator.
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
#linear stack of neural network layers
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
import pickle

(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),  32, 32, 3))
x_test = np.reshape(x_test, (len(x_test),  32, 32, 3))



########################
print("noise pickle loading")
file_Name_train = "cifar10_x_train_noise" 
file_Name_test = "cifar10_x_test_noise" 
fileObject = open(file_Name_train,'rb') 
x_train_noisy = pickle.load(fileObject)
print (x_train_noisy.shape)
fileObject = open(file_Name_test,'rb') 
x_test_noisy = pickle.load(fileObject)
print (x_test_noisy.shape)
fileObject.close()
print("noise pickle loading finished")
##################

input_img = Input(shape=(32,32,3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (32, 16, 16)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='Nadam', loss='binary_crossentropy')
history = autoencoder.fit(x_train_noisy, x_train,
                nb_epoch = 1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
                #callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])    
decoded_imgs = autoencoder.predict(x_test_noisy)
# use Matplotlib (don't ask)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.show()