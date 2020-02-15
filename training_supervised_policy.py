from __future__ import print_function
import constants
from random import sample
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import regularizers

import matplotlib.pylab as plt
import utils
import os

def perform_training(initializing, netname, numlayers = 6,
                     inpath = "SolvedLevels/", epochs = 3,
                     training_sets = 2):
    """

    Parameters
    ----------
    initializing : boolean
        Is True if the net already exists and we want to continue
        training and False if we want to make a new net.
    netname : string
        The name of the network in the file system.
    numlayers: int, optional
        Number of layers to use in the network. The default is 6.
    inpath : string, optional
        Location of levels. The default is "SolvedLevels/".
    epochs: int, optional
        Number of epochs to do per training set. The default is 3.
    training_sets: int, optional
        Number of training sets to sample from all possible data
        points. The default is 5.

    Returns
    -------
    The trained model

    """
    
    
    # Set up training and test data.  Inputs are positions,
    # outputs are (x,y,direction) tuples, representating
    # either a push or a win.
    #x_train, y_train = utils.load_levels(levels, inpath)
    x_test, y_test = utils.load_levels(constants.TEST_LEVELS, inpath)
    
    #utils.shuffle_in_unison(x_train, y_train)
    utils.shuffle_in_unison(x_test, y_test)
    
    # This line implicitly assumes that all levels have the same size.
    img_x, img_y, img_z = x_test[0].shape
    
    batch_size = 32
    num_classes = 4*img_x*img_y
    
    input_shape = (img_x, img_y, img_z)
    
    #x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    dconst = 0.3  # Dropout between hidden layers
    #dconst = 0.05
    # Should try regularizing instead
    
    model = None  # To give the variable global scope
    if initializing:
        model = Sequential()
        #model.add(Dropout(dconst, input_shape = input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape,
                         padding = 'same'))
                         #kernel_regularizer=regularizers.l2(.02), padding = 'same'))
        model.add(Dropout(dconst))
        #model.add(MaxPooling2D())
        for i in range(numlayers - 1):
            model.add(Conv2D(64, (3, 3), activation='relu',
                             padding = 'same'))
                             #kernel_regularizer=regularizers.l2(.02), padding = 'same'))
            model.add(Dropout(dconst))
        model.add(Flatten())
        #model.add(Dense(10, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    else:
        json_file = open("networks/policy_"+ netname +".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("networks/policy_" + netname + ".h5")
        print("Loaded model from disk")
        
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                optimizer = tensorflow.keras.optimizers.Adam(),
                metrics=['accuracy'])
    
    class AccuracyHistory(tensorflow.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []
    
        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))
    
    history = AccuracyHistory()
    
    for i in range(training_sets):
        levels_to_train = constants.TRAIN_LEVELS#sample(levels, 20)
        x_train, y_train = utils.load_levels(levels_to_train, inpath, shifts = True)
        #for check_over in range(len(x_train)):
        #    move = np.argmax(y_train[check_over])
        #    x_pos = move//80
        #    y_pos = (move%80)//4
        #    direction = move%4
        #    print(str(x_pos) + " " + str(y_pos) + " " + str(direction))
        #    if x_train[check_over, x_pos, y_pos, direction+6] == 0:
        #        print("Bad data!!!")
        utils.shuffle_in_unison(x_train, y_train)
        x_train = x_train.astype('float32')
        track = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks=[history])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, epochs+1), track.history['val_accuracy'])
    plt.plot(range(1, epochs+1), track.history['accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    
    model_json = model.to_json()
    dir = os.getcwd()+'/networks'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open("networks/policy_" + netname + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights("networks/policy_" + netname + ".h5")
    print("Saved model to disk")
    
    return model