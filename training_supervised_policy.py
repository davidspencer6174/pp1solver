# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:28:52 2018

@author: David
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, model_from_json
from keras import regularizers

import matplotlib.pylab as plt
import utils

# Option to either start training from scratch or
# load network and keep training.
initializing = True

inpath = "SolvedLevels/"


levels = ["Sandstorm", "Dividers", "AllLockedUp", "ClosetSpace",
          "LeakInTheRoof", "2112GrandFinale", "WetSalmon", "RushingRiver",
          "CupOfWaffles", "Abandoned", "RunningMan", "Thirteen",
          "ReturnPostage", "Lines", "LiquidSuspension", "PizzaCat",
          "Foundation", "SimpleIsGood", "Stripes", "SectorThree",
          "SilentButDeadly2", "BlocksOfCheese", "CampbellsChickenNoodleSoup",
          "TwoFaced", "Temptation", "OverMyHead", "MyLife", "Thanksgiving",
          "Breif", "Superman", "Landskrona", "CambodianHoliday",
          "YumayaYOchun", "Convalescence", "Moose", "Mosquera", "Qualot",
          "PimP", "Equinox", "Jussipekka", "Prodigal", "Persim",
          "DoubleConfusion", "Pomeg", "Watmel", "Aidomok", "Cuchulainn",
          "MaputoExpress", "Hondew", "GossipCache", "RottenCore",
          "IMadeThisLevelDuringSchool", "MalteseFalcon", "Serenity",
          "Contentment", "TinMan", "Forever", "Superlock", "LOLevel",
          "BurnDownTheMission", "10by10", "Quickie", "YouMayBeRight", "Boo",
          "Misdirection", "NigerianWeatherForecast"]
test_levels = ["K2xlgames", "Crazystylie", "LongWayHome", "Tetris",
               "AllBoxedUp", "TrickQuestion", "Corner2Corner", "HouseofGod",
               "RussianDoll", "Stars", "Mysterioso", "Brain", "Equality",
               "IndustrialBell", "OriginofSymmetry", "Puncture", "RabbitHole",
               "RoundtheBend", "StuckZipper", "Truancy", "TidalWave",
               "Enclosure", "Octopussy", "Checkmate", "Vortex", "TTotal",
               "SmallIntestines", "TrapDoor", "LinktoPast", "CrossEyed"]


# Set up training and test data.  Inputs are positions,
# outputs are (x,y,direction) tuples, representating
# either a push or a win.
x_train, y_train = utils.load_levels(levels, inpath)
x_test, y_test = utils.load_levels(test_levels, inpath)

# This line implicitly assumes that all levels have the same size.
img_x, img_y, img_z = x_train[0].shape

batch_size = 4
num_classes = 4*img_x*img_y
epochs = 4

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