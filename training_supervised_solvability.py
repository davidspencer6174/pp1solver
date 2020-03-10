from __future__ import print_function
import constants
from random import sample
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import regularizers

import matplotlib.pylab as plt
import utils
import os

def perform_solvability_training(initializing, netname, numlayers = 6, epochs = 3,
                                  training_sets = 2, batch_size = 32,
                                  learning_rate = .001):
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
    epochs: int, optional
        Number of epochs to do per training set. The default is 3.
    training_sets: int, optional
        Number of training sets to sample from all possible data
        points. The default is 5.
    learning_rate: float, optional
        Learning rate of the Adam optimizer. Default is .001.

    Returns
    -------
    The trained model

    """
    
    
    # Set up training and test data.  Inputs are positions,
    # outputs are (x,y,direction) tuples encoded to integers
    # and then to one-hot vectors, representing
    # either a push or a win.
    x_test, y_test = utils.load_solvability_data(constants.TEST_LEVELS)
    
    # This line implicitly assumes that all levels have the same size.
    # Therefore, small levels are padded with unmovables.
    img_x, img_y, img_z = x_test[0].shape
    
    input_shape = (img_x, img_y, img_z)
    
    x_test = x_test.astype('float32')
    print(x_test.shape[0], 'test samples')
    
    dconst = 0.3  # Dropout between hidden layers
    
    model = None  # To give the variable global scope
    if initializing:
        # Create a convolutional network with numlayers layers of 3 by 3
        # convolutions and a dense layer at the end.
        # Use batch normalization and regularization.
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape,
                         #padding = 'same'))
                         kernel_regularizer=regularizers.l2(.5), padding = 'same'))
        model.add(Dropout(dconst))
        
        for i in range(numlayers - 1):
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu',
                             #padding = 'same'))
                             kernel_regularizer=regularizers.l2(.5), padding = 'same'))
            model.add(Dropout(dconst))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Load the model and its weights
        json_file = open("networks/policy_"+ netname +".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("networks/policy_" + netname + ".h5")
        print("Loaded model from disk")
        
    model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
    
    # Keep track of the model's accuracy
    class AccuracyHistory(tensorflow.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []
    
        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))
    
    history = AccuracyHistory()
    
    # Use different training datasets by getting different random
    # samples from the shifts of the input data
    for i in range(training_sets):
        print("training set", i)
        levels_to_train = constants.TRAIN_LEVELS
        x_train, y_train = utils.load_solvability_data(levels_to_train, shifts = True)
        utils.shuffle_in_unison(x_train, y_train)
        x_train = x_train.astype('float32')
        
        # Train the network
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
    
    # Save the trained network
    model_json = model.to_json()
    directory = os.getcwd()+'/networks'
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open("networks/solvability_" + netname + ".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights("networks/solvability_" + netname + ".h5")
    print("Saved model to disk")
    
    return model