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


class PushPosition:
    """
    A PushPosition is a board position allowing the action
    space to be discretized by pushes.
    Discretizing by steps is also permitted for the sake of gameplay.
    In each case, penalties are step-wise, not push-wise.   
    The PushPosition is initialized with an array of shape
    (size, size, 12).  Only the first five layers need to be filled in
    when creating a PushPosition; other layers may be used by the neural network
    and are initialized and maintained by the class itself.
    
    Current layers:
    0: 1 for unmovable
    1: 1 for movable
    2: 1 for character
    3: 1 for winspace
    4: 1 for empty
    
    5: 1 for empty space accessible from current character location
    without having to make any pushes
    6: -#steps for movable which can be pushed in direction 0 (up) or,
    if a winspace is here and not covered by a movable, -#steps to
    reach the winspace.
    7: Same as 6 but for direction 1 (right)
    8: Same as 6 but for direction 2 (down)
    9: Same as 6 but for direction 3 (left)
    
    10: -#steps for exit reachable immediately
    11: number of times a square was empty and accessible.
    Note that the exit and character squares are considered empty.
    
    Layers 6:10 are required for internal maintenance of the stepcount,
    and so should not be modified without inserting some other method
    of keeping track of the steps. However, they aren't particularly
    suitable as inputs to the neural network, since -1 indicates a
    very cheap move, -50 indicates a very expensive move, and 0
    indicates an impossible move. This lack of monotonicity in the
    desirability of each move according to this input is probably harder
    to learn. Therefore, these layers are modified for the input to the
    neural network. This is done in position_transform.
    """
    
    def __init__(self, arr):
        """
        Initializes the position.  The first five planes define the
        level.  The remaining layers are ignored and recomputed.
        """
        # Allowing for adding input planes, in case we want to add more
        if arr.shape[2] < num_layers:
            new_arr = np.zeros((size, size, num_layers))
            new_arr[:,:,0:arr.shape[2]] = arr
            arr = new_arr
        self.size = arr.shape[1]
        self.arr = arr
        # Array to track moves (discretized by pushes)
        self.moves = []
        for x in range(self.size):  # Locate the character and exit
            for y in range(self.size):
                if arr[x, y, 2] == 1:
                    self.char_loc = [x, y]
                if arr[x, y, 3] == 1:
                    self.exit_loc = [x, y]
        # Clear the extra layers
        self.arr[:,:,5:] = np.zeros((self.size, self.size, num_layers-5))
        # Initialize the step counter
        self.steps = 0
        # Figure out which pushes are legal and which empty
        # squares can be reached
        self.assign_pushes()
        #self.orig_movables = copy.deepcopy(arr[:,:,1])
        
    def is_unmovable(self, x, y):
        return (self.arr[x, y, 0] == 1)
    
    def is_movable(self, x, y):
        return (self.arr[x, y, 1] == 1)
    
    def is_char(self, x, y):
        return (self.arr[x, y, 2] == 1)
        
    def is_win(self, x, y):
        return (self.arr[x, y, 3] == 1)
    
    def is_empty(self, x, y):
        return (self.arr[x, y, 4] == 1)
        
    def assign_pushes(self):
        """
        Only for internal use.
        Whenever the position is initialized or updated, call this to
        identify all available pushes from the current position.
        Does so via a breadth-first search, computing the smallest
        number of steps to reach each empty space and each push.
        Updates layers 5, 6, 7, 8, 9, 11
        """        
        self.arr[:,:,5:10] = np.zeros((self.size, self.size, 5))
        # Note that the character's current position is reachable
        self.arr[self.char_loc[0], self.char_loc[1], 5] = 1
        # Track the number of steps away from the character
        number_steps = 0
        # Track unexplored squares that need to be explored
        # (because they have been found to be reachable)
        squares = [self.char_loc]
        vecs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        while len(squares) > 0:
            number_steps += 1
            new_squares = []
            for square in squares:
                #For each neighbor, process it using the append_square
                #function.
                for move in range(4):
                    self.append_square(new_squares, square,
                                       vecs[move], move, number_steps)
            squares = new_squares
        self.arr[:,:,11] += self.arr[:,:,5]
        
    def in_bounds(self, x, y):
        return (x >= 0 and y >= 0 and x < self.size and y < self.size)
                    
    def make_move(self, x, y, direction, notifying_illegal = False,
                  force = False, assign_pushes = True):
        """
        Makes a pushwise move at (x,y) in one of the four directions.
        By default, this move will only occur if it is known to
        be legal due to an indicator at (x,y) in plane
        6, 7, 8, or 9 (depending on direction).
        However, in some situations we may wish to execute a sequence
        of moves which are already known to be legal without
        maintaining the planes for the sake of computational
        efficiency (such as during a tree search). In these cases,
        force=True overrides the legality check and assign_pushes=False
        preempts maintenance of those planes.
        Note that this will result in inaccurate stepcounts, so those
        should still be maintained somehow.
        """
        if (not self.in_bounds(x, y) or
            self.arr[x, y, direction+6] == 0 and not force):
            if notifying_illegal:
                print("Illegal move")
            return constants.ILLEGAL
        vecs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        # Add this move to the list of moves.
        self.moves.append(np.array([x, y, direction]))
        # Check whether a win was reached without pushing
        if self.is_win(x, y) and not self.is_movable(x, y):
            self.steps -= self.arr[x, y, 10]
            return constants.WIN  # Should not play more moves after this
        self.steps -= self.arr[x, y, direction+6]
        # Check whether a win was reached by pushing
        if self.is_win(x, y):
            return constants.WIN
        # Three planes to update: character, movable, empty.
        self.arr[self.char_loc[0], self.char_loc[1], 2] = 0
        self.arr[x, y, 1] = 0
        self.arr[x, y, 2] = 1
        self.arr[x, y, 4] = 1
        self.arr[x+vecs[direction][0], y+vecs[direction][1], 4] = 0
        self.arr[x+vecs[direction][0], y+vecs[direction][1], 1] = 1
        self.char_loc = [x, y]
        # Maintain the possible pushes plane
        if assign_pushes:
            self.assign_pushes()
        return constants.PUSH
        
    def make_move_number(self, move):
        """
        Handles decoding for move numbers used by net.
        The move is encoded using three variables: x position,
        y position, and the direction of the push.
        """
        x = move//(self.size*4)
        y = move%(self.size*4)//4
        direction = move%4
        return self.make_move(x, y, direction)
    
    def get_legal_move_numbers(self):
        """Gets a numpy array of legal move
        numbers"""
        out = np.nonzero(self.arr[:,:,6:10].flatten())[0]
        for move in out:
            copy_pos = PushPosition(copy.deepcopy(self.arr))
            assert copy_pos.make_move_number(move) != constants.ILLEGAL
            
        return out

    def step_in_direction(self, direction):
        """
        Steps in a direction for the sake of gameplay.
        Output is true if and only if the step is legal.
        """
        vec = [[-1, 0], [0, 1], [1, 0], [0, -1]][direction]
        # Determine the character's new position
        new_x = self.char_loc[0] + vec[0]
        new_y = self.char_loc[1] + vec[1]
        # Stepping out of bounds is illegal.
        if not self.in_bounds(new_x, new_y):
            return False
        # Check whether the requested step is a legal push or win
        # using the already-computed push planes. If not, need
        # more work.
        if self.arr[new_x, new_y, direction+6] == 0:
            # If the requested step hits something,
            if (self.is_unmovable(new_x, new_y)
                or self.is_movable(new_x, new_y)):
                return False
            # The step is now known to be legal (and it is in
            # empty space, since it's not a push or win).
            # Move the character
            self.arr[self.char_loc[0], self.char_loc[1], 2] = 0
            self.arr[new_x, new_y, 2] = 1
            self.char_loc = [new_x, new_y]
            # Now need to redo planes with new distances
            self.assign_pushes()
            self.steps += 1
            return True
        # If the requested step is a legal push or win, can
        # use the make_move function.
        self.steps += 1
        self.make_move(new_x, new_y, direction)
        return True
        
                    
    def append_square(self, new_squares, square, vec, move, num_moves):
        """
        Only for internal use.
        Checks whether the square can be reached.
        If it is empty, add it to plane 5.
        If it contains a block that can be pushed or an exit, add
        it to planes 6-9 inclusive as appropriate.
        Maintains the new_squares list from its caller.
        """
        x = square[0] + vec[0]
        y = square[1] + vec[1]
        # Out of bounds
        if not self.in_bounds(x, y):
            return
        # Empty space with no winspace. Need to investigate
        # this square further.
        if self.is_empty(x, y) and not self.is_win(x, y):
            if self.arr[x, y, 5] == 1:  # Already explored
                return
            self.arr[x, y, 5] = 1
            new_squares.append([x, y])  # Need to explore this square
            return
        # Winspace
        if self.is_win(x, y) and self.is_empty(x, y):
            # Already entered winspace at least as quickly
            if np.sum(self.arr[x, y, 6:10]) != 0:
                return
            # Newly entering winspace and valid to enter
            # No associated orientation.  Also update plane 10.
            self.arr[x, y, move+6] = -num_moves
            self.arr[x, y, 10] = -num_moves
            return
        if not self.in_bounds(x+vec[0], y+vec[1]):
            return
        # The only remaining legal case is a pushable movable.
        if self.is_movable(x, y) and self.is_empty(x+vec[0], y+vec[1]):
            self.arr[x, y, move+6] = -num_moves
            # Now account for possible win
            # It may be possible to push from multiple directions.
            # We want the shorter one, so take the max.
            if self.is_win(x, y):
                self.arr[x, y, 10] = max(self.arr[x, y, 10], -num_moves)
            
    def prettystring(self):
        """Prints the board in a somewhat human-readable format"""
        out = ""
        for i in range(self.size):
            for j in range(self.size):
                if self.arr[i,j,0] == 1:
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
    
    
def append_solvability_data(position, solvable, data_x, data_y, shifts = False):
    rng_seq = np.random.rand(20000)
    rng_indices = [0, 0]
    
    x_shifts(copy.deepcopy(position.arr), data_x, rng_seq,
             rng_indices, shifts = shifts)
    y_solvability_shifts(solvable, data_y, position.arr,
                         rng_seq, rng_indices, shifts = shifts)

        
def append_level_push_data(file_string, data_x, data_y, shifts = False):
    """
    file_string: filename for the level
    data_x: list of input positions to which data
    for this level should be appended
    data_y: list of correct moves to which correct
    moves for this level should be appended
    shifts: whether to include rotations and
    translations of the positions
    """
    f = open(file_string, "rb")
    lvl_data = pickle.load(f)
    # lvl_data contains three things: an array that can be used to
    # initialize the starting position, an array with the character
    # location, and the sequence of steps that beats the level.
    f.close()
    
    # Including translations, there is too much data to hold
    # in memory. Only keep some, but make sure
    # to keep the same x (input) and y (output) data points.
    # Accomplish this by using the same sequence of random numbers
    # for x and y.
    rng_seq = np.random.rand(20000)
    rng_indices = [0, 0]
    
    arr, char_loc, steps = np.array(lvl_data[0]), lvl_data[1], lvl_data[2]
    # This data is stepwise (since it was produced by user gameplay).
    # Need to convert it to pushwise data.
    size = arr.shape[0]
    vecs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    push_pos = PushPosition(arr)
    # Add x-data for the starting position. The solution's push
    # is not known yet, but when it is known, it will be added to y-data.
    x_shifts(copy.deepcopy(push_pos.arr), data_x,
             rng_seq, rng_indices, shifts = shifts)
    prev_char_x, prev_char_y = push_pos.char_loc
    before_augmenting = 0
    # Go through all of the steps in the solution, saving y-data each
    # time we make a push or solve the level and then saving x-data.
    for (ind, step) in enumerate(steps):
        # Take the step
        vec = vecs[step]
        new_char_x = push_pos.char_loc[0]+vec[0]
        new_char_y = push_pos.char_loc[1]+vec[1]
        # Determine whether this step resulted in a move
        # (meaning either a push or a win)
        if (push_pos.is_movable(new_char_x, new_char_y)
            or push_pos.is_win(new_char_x, new_char_y)):
            # Save y-data which should be associated with the most
            # recently saved x-data.
            y_push_shifts(new_char_x, new_char_y, step, data_y, push_pos.arr,
                     rng_seq, rng_indices, shifts = shifts)
            # If the step is illegal, the data is bad, since this
            # is supposed to be an optimal solution.
            if not push_pos.step_in_direction(step):
                print("Level did not load properly.")
            # If we have reached a win, don't want to keep the
            # x-data we are about to save, since there's no
            # associated y-data. Store an index so we can
            # delete it later if needed.
            before_augmenting = len(data_x)
            # Save x-data
            x_shifts(copy.deepcopy(push_pos.arr), data_x,
                     rng_seq, rng_indices, shifts = shifts)
            prev_char_x, prev_char_y = new_char_x, new_char_y
        else:
            if not push_pos.step_in_direction(step):
                print("Level did not load properly.")
    # Get rid of x-data which is not associated with any y-data
    del data_x[before_augmenting:]
    # Print out the length of the input data and output
    # data, which should always be the same
    print(len(data_x))
    print(len(data_y))

            
def x_rotations(arr, data_x):
    """
    Computes all rotations of an input board and appends
    them to data_x.
    """
    # Make a copy of the array and transform it to prepare to
    # send it to the neural network.
    cop_arr = copy.deepcopy(arr)
    position_transform(cop_arr)
    # Save and rotate the position four times
    for i in range(4):
        data_x.append(copy.deepcopy(cop_arr))
        cop_arr = np.rot90(cop_arr)  # Rotate
        # Now have to fix pushes, since what used to be up
        # is no longer up.
        cop_arr[:,:,6:10] = np.roll(cop_arr[:,:,6:10], -1, axis=2)
    # Now reflect it for the other four possibilities
    cop_arr = cop_arr.swapaxes(0, 1)
    cop_arr[:,:,6:10] = np.flip(cop_arr[:,:,6:10], 2)
    for i in range(4):
        data_x.append(copy.deepcopy(cop_arr))
        cop_arr = np.rot90(cop_arr)
        cop_arr[:,:,6:10] = np.roll(cop_arr[:,:,6:10], -1, axis=2)
    cop_arr[:,:,6:10] = np.flip(cop_arr[:,:,6:10], 2)
    cop_arr = cop_arr.swapaxes(0, 1)
    
def x_shifts(arr, data_x, rng_seq, rng_indices, shifts = False):
    """
    Computes all valid shifts of an input board arr and all
    rotations of those shifts and appends them to data_x.
    rng_seq and rng_indices are used to ensure that the same
    data are included for x-shifts and y-shifts.
    """
    # Compute the width and the height of the level to figure
    # out how much we can translate it without messing anything up
    width, height = tuple(coords.max() for coords in np.nonzero(1 - arr[:,:,0]))
    if not shifts:
        width = 19
        height = 19
    # Enumerate possible shifts
    for shift_x in range(20 - width):
        for shift_y in range(20 - height):
            # Roll the array in each axis
            shifted_arr_x = np.roll(arr, shift_x, axis = 0)
            shifted_arr_both = np.roll(shifted_arr_x, shift_y, axis = 1)
            # Decide whether to keep data for this shift
            if rng_seq[rng_indices[0]] < constants.EXPECTED_SHIFTS_CHOSEN/((20-width)*(20-height)):
                x_rotations(copy.deepcopy(shifted_arr_both), data_x)
            rng_indices[0] += 1


def rotate_once(x, y, direction):
    """
    Given coordinates and a direction, computes what
    they will be after a rotation.
    """
    size = constants.SIZE
    return size - y - 1, x, (direction-1) % 4

        
def y_push_rotations(x, y, direction, data_y):
    """
    Transforms move numbers to match the rotations of the
    input board and appends them to data_y.
    """
    size = constants.SIZE
    # Do each of four rotations
    for i in range(4):
        # Add encoded move
        data_y.append(onehot(size*size*4,
                             4*size*x + 4*y + direction))
        x, y, direction = rotate_once(x, y, direction)
    direction = 3 - direction
    # Reflect and then do each of four rotations again
    x, y = y, x
    for i in range(4):
        data_y.append(onehot(size*size*4,
                             4*size*x + 4*y + direction))
        x, y, direction = rotate_once(x, y, direction)
    x, y = y, x
    direction = 3 - direction
    
def y_push_shifts(move_x, move_y, direction, data_y, arr, rng_seq, rng_indices, shifts = False):
    """
    Transforms move numbers to match the shifts and rotations
    of the input board and appends them to data_y.
    """
    # Compute the width and the height of the level to figure
    # out how much we can translate it without messing anything up
    width, height = tuple(coords.max() for coords in np.nonzero(1 - arr[:,:,0]))
    if not shifts:
        width = 19
        height = 19
    # Enumerate possible shifts
    for shift_x in range(20 - width):
        for shift_y in range(20 - height):
            # Decide whether to keep data for this shift
            if rng_seq[rng_indices[1]] < constants.EXPECTED_SHIFTS_CHOSEN/((20-width)*(20-height)):
                # Append data shifted in each axis
                y_push_rotations(move_x+shift_x, move_y+shift_y, direction, data_y)
            rng_indices[1] += 1
            
def y_solvability_shifts(solvable, data_y, arr, rng_seq, rng_indices, shifts = False):
    width, height = tuple(coords.max() for coords in np.nonzero(1 - arr[:,:,0]))
    if not shifts:
        width = 19
        height = 19
    for shift_x in range(20 - width):
        for shift_y in range(20 - height):
            if rng_seq[rng_indices[1]] < constants.EXPECTED_SHIFTS_CHOSEN/((20-width)*(20-height)):
                for rotreflect in range(8):
                    data_y.append(np.array([solvable]))
            rng_indices[1] += 1


    

    
def import_raw_level(level):
    """
    Imports a level given the level code in the official PP2 format.
    Pads the level with unmovables to make it 20x20.
    """
    f = open(constants.RAWPATH+level+".txt", "r")
    level = f.readline().split()  
    metadata = level[0].split(",")
    width, height = int(metadata[2]), int(metadata[3])
    arr = np.zeros((20, 20, 10))
    arr[:,:,4] = 1
    for i in range(0, 20):
        for j in range(0, 20):
            if i >= height or j >= width:
                arr[i,j,0] = 1
                arr[i,j,4] = 0
    for item in level[1:]:
        vals = item.split(":")[1].split(",")
        if item[0] == "w":
            arr[int(vals[1]),int(vals[0]),3] = 1
        if item[0] == "b":
            arr[int(vals[1]),int(vals[0]),0] = 1
            arr[int(vals[1]),int(vals[0]),4] = 0
        if item[0] == "m":
            arr[int(vals[1]),int(vals[0]),1] = 1
            arr[int(vals[1]),int(vals[0]),4] = 0
        if item[0] == "c":
            arr[int(vals[1]),int(vals[0]),2] = 1
    p = PushPosition(arr)
    return p, width, height

def encode(x, y, direction):
    return direction + y*4 + x*4*constants.SIZE

        
def onehot(length, ind):
    out = np.zeros((length))
    out[ind] = 1
    return out

    
def get_position(arr, moves):
    """
    Fetches a PushPosition based on the initial array and the
    moves list.  Moves are discretized by pushes.
    """
    # Create a PushPosition from the original array
    p = PushPosition(arr)
    # Make each move in the moves list
    for move in moves:
        p.make_move(move[0], move[1], move[2], notifying_illegal = True)
    return p

def get_position_faster(arr, moves, final_stepcount):
    """
    Should only be used if the moves are known to be
    legal and the final stepcount is known. Saves computational
    time by avoiding maintenance of planes
    """
    p = PushPosition(arr)
    for (i, move) in enumerate(moves):
        # We only need to maintain the planes (by calling
        # assign_pushes) if we are on the last move.
        p.make_move(move[0], move[1], move[2], force = True,
                    assign_pushes = (i == len(moves) - 1))
    p.steps = final_stepcount
    return p
    
def set_up_position(pass_in, size_x, size_y): #setting up a position given an array of 0, 1, 2, etc.
    arr = np.zeros((20, 20, num_layers))
    arr[:,:,0] = np.ones((20, 20))
    arr[:size_x, :size_y, 0] = np.zeros((size_x, size_y))
    for x in range(size_x):
        for y in range(size_y):
            if pass_in[x, y] < 5: #want to allow both movable and exit
                arr[x,y,pass_in[x, y]] = 1
            else:
                arr[x,y,1] = 1
                arr[x,y,3] = 1
    return PushPosition(arr)

def make_curriculum_pos(position, proportion):
    new_arr = copy.deepcopy(position.arr)
    new_arr[:,:,1] = (new_arr[:,:,1] *
                      np.random.binomial(1,
                                         proportion,
                                         size = (constants.SIZE, constants.SIZE)))
    new_arr[:,:,4] += position.arr[:,:,1] - new_arr[:,:,1]
    return PushPosition(new_arr)


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

def initialize_model(numlayers, learning_rate):
    num_classes = 4*constants.SIZE*constants.SIZE   
    
    inp = Input((constants.SIZE,
                 constants.SIZE,
                 12))
    for i in range(numlayers):
        x = Conv2D(64, (3, 3), activation = 'relu')(inp)
    x = Flatten()(x)
    out1 = Dense(num_classes, activation='softmax')(x)
    out2 = Dense(1, activation='tanh')(x)
    model = Model(inp, [out1, out2])
    model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss = [tensorflow.keras.losses.categorical_crossentropy,
                          tensorflow.keras.losses.MSE])
                          #tensorflow.keras.losses.binary_crossentropy])
    return model
    
    #model.fit(inputData,[outputYLeft, outputYRight], epochs=..., batch_size=...)