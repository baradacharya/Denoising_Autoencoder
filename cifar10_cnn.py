# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:30:26 2017

@author: 7
"""

from __future__ import print_function
import numpy as np
#setting a seed for the computer's pseudorandom number generator.
np.random.seed(1337)  # for reproducibility
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
from scipy import misc
import matplotlib.pyplot as plt



map = dict()
map[0] = "plane" 
map[1] = "automob" 
map[2] = "bird" 
map[3] = "cat" 
map[4] = "deer"
map[5] = "dog" 
map[6] = "frog" 
map[7] = "horse" 
map[8] = "ship" 
map[9] = "truck"


batch_size = 32
nb_classes = 10
nb_epoch = 1
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('X_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
print(x_test.shape, 'test samples')
print(y_test.shape, 'y test samples')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 3)

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
#(None,32, 32, 32)
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1]))
#(None,30, 30, 32)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
#(None,15, 15, 32)
model.add(Dropout(0.5))


model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1],border_mode='same'))

#(None,15,15,32)
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1]))

#(None,13, 13, 32)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
#(None,6, 6, 32)
model.add(Dropout(0.25))
model.add(Flatten())
#(None,1152)
model.add(Dense(512))
#(None,512)
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
#(None,10)
model.summary()
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(x_test, y_test),
              shuffle=True)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted_classes = model.predict_classes(x_test)
correct_indices = np.nonzero(predicted_classes == y_test.argmax(axis=-1))[0]
incorrect_indices = np.nonzero(predicted_classes != y_test.argmax(axis=-1))[0]

plt.figure(1, figsize=(7,7))

for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(32,32,3))
    plt.title("Pred {}, class {}".format(map[predicted_classes[correct]], map[y_test[correct].argmax(axis=-1)]))
    plt.xticks([])
    plt.yticks([])

plt.figure(2, figsize=(7,7))
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(32,32,3))
    plt.title("Pred {}, Class {}".format(map[predicted_classes[incorrect]], map[y_test[incorrect].argmax(axis=-1)]))
    plt.xticks([])
    plt.yticks([])
# Plot loss trajectory throughout training.    
plt.show()
plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()