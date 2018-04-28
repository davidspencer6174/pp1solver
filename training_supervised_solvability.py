# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:28:52 2018

@author: David
"""

from __future__ import print_function
import keras
import pickle
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, model_from_json
from keras import regularizers
import numpy as np

import matplotlib.pylab as plt
import utils

def load_levels(levels, path):
    x_data = []
    y_data = []
    for level in levels:
        print(level)
        f = open(path+level+"_altered", "rb")
        lvl_data = pickle.load(f)
        for pos in range(len(lvl_data)//2):
            arr = np.array(lvl_data[pos*2])
            utils.x_rotations(arr, x_data)
            for i in range(8):
                if lvl_data[pos*2+1] == 'y':
                    y_data.append(np.array([0, 1]))
                else:
                    y_data.append(np.array([1, 0]))
        f.close()
    return np.array(x_data), np.array(y_data)
        

# Option to either start training from scratch or
# load network and keep training.
initializing = True

inpath = "AlteredLevels/"


levels = ["9by8", "2112GrandFinale", "Abandoned", "Aidomok", "AllLockedUp",
          "Brain", "BurnDownTheMission", "DuelCarriageway1", "GossipCache"]

test_levels = ["Watmel"]


# Set up training and test data.  Inputs are positions,
# outputs are (x,y,direction) tuples, representating
# either a push or a win.
x_train, y_train = load_levels(levels, inpath)
x_test, y_test = load_levels(test_levels, inpath)

# This line implicitly assumes that all levels have the same size.
img_x, img_y, img_z = x_train[0].shape

batch_size = 4
num_classes = 2
epochs = 10

input_shape = (img_x, img_y, img_z)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

dconst = 0.3  # Dropout between hidden layers
# Should try regularizing instead

model = None  # To give the variable global scope
if initializing:
    model = Sequential()
    #model.add(Dropout(dconst, input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape))
    model.add(Dropout(dconst))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(dconst))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(Dropout(dconst))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(Dropout(dconst))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
else:
    json_file = open("networks/policy.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("networks/policy.h5")
    print("Loaded model from disk")
    
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer = keras.optimizers.Adam(),
            metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

track = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, epochs+1), track.history['val_acc'])
plt.plot(range(1, epochs+1), track.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

model_json = model.to_json()
with open("networks/policy.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("networks/policy.h5")
print("Saved model to disk")