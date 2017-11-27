# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 02:47:50 2017

@author: 7
"""
from __future__ import print_function
import numpy as np
#setting a seed for the computer's pseudorandom number generator.
np.random.seed(1337)  # for reproducibility

#linear stack of neural network layers
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from Image_Loader import LoadImages
from Image_Saver import SaveImages
from keras.models import Sequential

#################################
batch_size = 1
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 420, 540
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)



#############################
path1 = "train_modified\\"
imgs = LoadImages(path1)
x_train = np.array( imgs )
print (x_train.shape)
path1 = "test_modified\\"
imgs = LoadImages(path1)
x_test = np.array( imgs )
print (x_test.shape)
path1 = "train_cleaned_modified\\"
imgs = LoadImages(path1)
x_train_cleaned = np.array( imgs )
print (x_train_cleaned.shape)

path1 = "train_valid\\"
imgs = LoadImages(path1)
x_val = np.array( imgs )
print (x_val.shape)

path1 = "train_valid_cleaned\\"
imgs = LoadImages(path1)
x_val_cleaned = np.array( imgs )
print (x_val_cleaned.shape)

x_train = x_train.astype('float32') / 255.
x_train_cleaned = x_train_cleaned.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_val_cleaned = x_val_cleaned.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
                   
x_train = np.reshape(x_train, (len(x_train), img_rows, img_cols,1))
x_test = np.reshape(x_test, (len(x_test), img_rows, img_cols,1))
x_train_cleaned = np.reshape(x_train_cleaned, (len(x_train_cleaned), img_rows, img_cols,1))
x_val_cleaned = np.reshape(x_val_cleaned, (len(x_val_cleaned), img_rows, img_cols,1))
x_val = np.reshape(x_val, (len(x_val), img_rows, img_cols,1))
input_img = Input(shape=(img_rows, img_cols,1))
#input_shape = ( img_rows, img_cols , 1)
x = Convolution2D(nb_filters,  kernel_size[0], kernel_size[1], activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(nb_filters, 5, 5, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (32, 16, 16)

x = Convolution2D(nb_filters,  kernel_size[0], kernel_size[1], activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Convolution2D(nb_filters,  kernel_size[0], kernel_size[1], activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, kernel_size[0], kernel_size[1], activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()


autoencoder.compile(optimizer='Nadam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train_cleaned,
                nb_epoch=nb_epoch,
                batch_size=1,
                shuffle=True,
                validation_data=(x_val, x_val_cleaned))
                  
decoded_imgs = autoencoder.predict(x_test,batch_size=1)

for i in range(x_test.shape[0]):
    path2 = "test_clean\\" + str(i)
    SaveImages(decoded_imgs[i].reshape(420, 540),path2)     
    
plt.show()

plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.show()
