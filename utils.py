import pickle
import copy
import random

import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import tensorflow.keras
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import regularizers

import numpy as np
import constants

num_layers = 12
size = 20

class StepPosition:
    """
    A StepPosition is a board position discretizing the action
    space by steps. It is much cheaper to compute updates for
    such a board, but the tradeoff is that solutions become
    much longer and the neural net has to plan more standard
    types of motions.
    
    Layers:
    0: 0 for unmovable (so that 0 padding gives unmovables)
    1: 1 for movable
    2: 1 for character
    3: 1 for winspace
    4: 1 for empty
    """
    def __init__(self, arr):
        """
        Initializes the position.  The first five planes define the
        level.  The remaining layers are ignored and recomputed.
        """
        self.size = arr.shape[1]
        self.arr = arr
        # Array to track stepmoves (discretized by U, L, D, R)
        self.moves = []
        #Network is having trouble learning to go to the
        #exit. Hack to force it not to retrace its path without
        #doing a push
        for x in range(self.size):  # Locate the character and exit
            for y in range(self.size):
                if arr[x, y, 2] == 1:
                    self.char_loc = (x, y)
                if arr[x, y, 3] == 1:
                    self.exit_loc = (x, y)
        # Initialize the step counter
        self.steps = 0
        
    def is_unmovable(self, x, y):
        return (self.arr[x, y, 0] == 0)
    
    def is_movable(self, x, y):
        return (self.arr[x, y, 1] == 1)
    
    def is_char(self, x, y):
        return (self.arr[x, y, 2] == 1)
        
    def is_win(self, x, y):
        return (self.arr[x, y, 3] == 1)
    
    def is_empty(self, x, y):
        return (self.arr[x, y, 4] == 1)
    
    def is_visited_since_push(self, x, y):
        return (self.arr[x, y, 5] == 1)
        
    def in_bounds(self, x, y):
        return (x >= 0 and y >= 0 and x < self.size and y < self.size)
                    
    def make_move(self, direction, actually_move = True):
        """
        Makes a stepwise move in one of the four directions.
        If actually_move, the planes get updated.
        Can use actually_move = False to test legality
        of a move.
        """
        vecs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        vec = vecs[direction]
        new_x = self.char_loc[0] + vec[0]
        new_y = self.char_loc[1] + vec[1]
        if (not self.in_bounds(new_x, new_y) or
            self.is_unmovable(new_x, new_y) or
            self.is_visited_since_push(new_x, new_y)):
            return constants.ILLEGAL
        if self.is_movable(new_x, new_y):
            next_x = new_x + vec[0]
            next_y = new_y + vec[1]
            if (not self.in_bounds(next_x, next_y) or
                self.is_unmovable(next_x, next_y) or
                self.is_movable(next_x, next_y)):
                return constants.ILLEGAL
            if actually_move:
                self.steps += 1
                self.moves.append(direction)
            if self.is_win(new_x, new_y):
                return constants.WIN
            if actually_move:
                self.arr[self.char_loc[0], self.char_loc[1], 2] = 0
                # Three planes to update: character, movable, empty.
                self.arr[new_x, new_y, 1] = 0
                self.arr[new_x, new_y, 2] = 1
                self.arr[new_x, new_y, 4] = 1
                self.arr[next_x, next_y, 4] = 0
                self.arr[next_x, next_y, 1] = 1
                self.char_loc = (new_x, new_y)
                self.arr[:,:,5] = copy.deepcopy(self.arr[:,:,2])
            return constants.PUSH
        if actually_move:
            self.steps += 1
            self.moves.append(direction)
        if self.is_win(new_x, new_y):
            return constants.WIN
        if actually_move:
            self.arr[self.char_loc[0], self.char_loc[1], 2] = 0
            self.arr[new_x, new_y, 2] = 1
            self.char_loc = (new_x, new_y)
            self.arr[self.char_loc[0], self.char_loc[1], 5] = 1
        return constants.NORMAL
    
    def make_move_number(self, direction, actually_move = True):
        return self.make_move(direction, actually_move)
    
    def get_legal_move_numbers(self):
        """Gets a numpy array of legal move
        numbers"""
        out = []
        for direction in range(4):
            if self.make_move(direction, actually_move = False) != constants.ILLEGAL:
                out.append(direction)
        return out
    
    def get_hash(self):
        return hash(self.arr[:,:,0:3].tostring())
            
    def prettystring(self):
        """Prints the board in a somewhat human-readable format"""
        out = ""
        for i in range(self.size):
            for j in range(self.size):
                if self.arr[i,j,0] == 0:
                    out += "U"
                elif self.arr[i,j,1] == 1:
                    out += "M"
                    if self.arr[i,j,3] == 1:
                        out += "W"
                elif self.arr[i,j,2] == 1:
                    out += "C"
                elif self.arr[i,j,3] == 1:
                    out += "W"
                elif self.arr[i,j,4] == 1:
                    out += "-"
            out += "\n"
        return out


    

    
def import_raw_level(level):
    """
    Imports a level given the level code in the official PP2 format.
    Pads the level with unmovables to make it 20x20.
    Makes a StepPosition
    Note that a 0 in the unmovable plane represents an unmovable
    """
    f = open(constants.RAWPATH+level+".txt", "r")
    level = f.readline().split()  
    metadata = level[0].split(",")
    width, height = int(metadata[2]), int(metadata[3])
    inp_layers = 6
    arr = np.zeros((20, 20, inp_layers), dtype = bool)
    arr[:,:,0] = 1
    arr[:,:,4] = 1
    for i in range(0, 20):
        for j in range(0, 20):
            if i >= height or j >= width:
                arr[i,j,0] = 0
                arr[i,j,4] = 0
    for item in level[1:]:
        vals = item.split(":")[1].split(",")
        if item[0] == "w":
            arr[int(vals[1]),int(vals[0]),3] = 1
        if item[0] == "b":
            arr[int(vals[1]),int(vals[0]),0] = 0
            arr[int(vals[1]),int(vals[0]),4] = 0
        if item[0] == "m":
            arr[int(vals[1]),int(vals[0]),1] = 1
            arr[int(vals[1]),int(vals[0]),4] = 0
        if item[0] == "c":
            arr[int(vals[1]),int(vals[0]),2] = 1
    p = None
    arr[:,:,5] = copy.deepcopy(arr[:,:,2])
    p = StepPosition(arr)
    return p, width, height

def encode(x, y, direction):
    return direction + y*4 + x*4*constants.SIZE
        
def onehot(length, ind):
    out = np.zeros((length))
    out[ind] = 1
    return out

def make_curriculum_pos(position, proportion, random_translation = True,
                        random_rot_flip = True):
    new_arr = copy.deepcopy(position.arr)
    new_arr[:,:,1] = (new_arr[:,:,1] *
                      np.random.binomial(1,
                                         proportion,
                                         size = (constants.SIZE, constants.SIZE)))
    new_arr[:,:,4] = new_arr[:,:,4] | (position.arr[:,:,1] ^ new_arr[:,:,1])
    if random_translation:
        width, height = tuple(coords.max() for coords in np.nonzero(1 - new_arr[:,:,0]))
        shift_x = random.randint(0, 20 - width - 1)
        shift_y = random.randint(0, 20 - height - 1)
        shifted_arr_x = np.roll(new_arr, shift_x, axis = 0)
        new_arr = np.roll(shifted_arr_x, shift_y, axis = 1)
    if random_rot_flip:
        if random.randint(0, 1) == 1:
            new_arr = new_arr.swapaxes(0, 1)
        for i in range(random.randint(0, 1)):
            new_arr = np.rot90(new_arr)
    return StepPosition(new_arr)


def shuffle_in_unison(l1, l2):
    """
    Used to shuffle training data.
    """
    indices = np.arange(l1.shape[0])
    np.random.shuffle(indices)
    l1 = l1[indices]
    l2 = l2[indices]

    
def load_model(netname):
    netpath = "networks/policy_" + str(netname) + ".json"
    weights = "networks/policy_" + str(netname) + ".h5"
    json_file = open(netpath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    return model

def initialize_model(numlayers):
    num_classes = 4
    
    inp_layers = 6
    #tensorflow.keras.backend.set_image_dim_ordering('th')
        
    inp = Input((constants.SIZE,
                 constants.SIZE,
                 inp_layers))
    x = Conv2D(64, (3, 3), activation = 'relu', padding='same')(inp)
    x = BatchNormalization()(x)
    for i in range(numlayers - 1):
        x = Conv2D(64, (3, 3), activation = 'relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = BatchNormalization()(x)
    out1 = Dense(num_classes, activation='softmax', name = 'out1')(x)
    out2 = Dense(1, activation='tanh', name = 'out2')(x)
    model = Model(inp, [out1, out2])
    model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=constants.LR),
                  loss = [tensorflow.keras.losses.categorical_crossentropy,
                          tensorflow.keras.losses.MSE],
                  loss_weights = [1, 30])
                          #tensorflow.keras.losses.binary_crossentropy])
    return model
    
    #model.fit(inputData,[outputYLeft, outputYRight], epochs=..., batch_size=...)
    
def report_training(training_data, index):
    print('Position:')
    print(StepPosition(training_data[0][index]).prettystring())
    print('Expected move probs: ' + str(training_data[1][index]))
    print('Score: ' + str(training_data[2][index]))