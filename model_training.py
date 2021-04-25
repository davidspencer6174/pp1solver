import numpy as np

def train_model(model, training_data):
    '''training_data is a tuple of lists:
    ([arrays from pushpositions for training
      positions],
     [probabilities of actions resulting from
      MCTS],
     [number of steps remaining before solving])'''
    batch_size = 32
    epochs = 2
    model.fit(np.array(training_data[0]),
              [np.array(training_data[1]),
               np.array(training_data[2])],
               batch_size=batch_size,
               epochs=epochs,
               verbose=1
               #validation_data=(x_test, y_test),
               #callbacks=[history]
               )