# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:27:26 2017

@author: 7
"""
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np

"""
    We haave to create a autoencoder with one hidden layer like input-->encoded-->decoded(original) 
    Steps:
        1. define a input layer of 784 dimenssion of 10 input numbers. (28*28 =784)
        2. create a compressed encoded layer of 32-dimenssion
        3. create a decompressed decoded layer of 784-dimenssion. it should represent original layer
        4. by combinig these 3-layers create an autoencoder and train it with same input and output
        5. Then using the autoencoder predict the output and plot both input and predicted output 

"""

'''#Step 1: preprocess the data'''
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train.shape = (60000, 28, 28)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_train.shape = (60000, 784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


"""#step :2. Configure the model"""
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

"""#Step 3: Compile model."""
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

"""#Step 4: train model."""
#train auto encoder with same input and output value
autoencoder.fit(x_train, x_train,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
