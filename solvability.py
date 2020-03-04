# We want to train a network that can estimate the
# probability that the position is solvable (i.e. that
# an exit can be reached from it).
# The SolvabilityData folder will be populated with pickle files,
# each of which contains data for one level and whose name
# corresponds to the level name.
# These pickle files will be filled with n two-tuples, where n is the
# number of data points for that level. Each two-tuple has as its first
# entry a sequence of moves in the level (expressed as three-tuples)
# and as its second entry either 0 (unsolvable) or 1 (solvable).
# From the sequence of moves and the level name, we can recover a position.

# We know that, for each level, each position during the true path is
# solvable. So we initialize each file with a bunch of solvable data.
# To create other data, do partial Monte Carlo rollouts, fixing a
# position p several moves deviated from the true path. Then do full
# rollouts from that position and store the moves that led to p along
# with whether a solution is found.

import constants
import copy
import montecarlo
import os
import pickle
import random
import utils

def add_data_point(lvl, moves, is_solvable, num_steps = None):
    try:
        f = open(constants.SOLVABILITYPATH + lvl, "rb")
        solvability_data = pickle.load(f)
        f.close()
        already_in = False
        soln_already_known = False
        # Check whether this point is already represented.
        for data_point in solvability_data:
            if [i for s in moves for i in s] == [i for s in data_point[0] for i in s]:
                already_in = True
                soln_already_known = data_point[1]
                data_point[1] = soln_already_known or is_solvable
                if soln_already_known:
                    if num_steps == None:
                        print("Already existing data")
                        return
                    num_steps = min(num_steps, data_point[2])
                data_point[2] = num_steps
                break
        if not already_in:
            solvability_data.append([moves, is_solvable, num_steps])
        f = open(constants.SOLVABILITYPATH + lvl, "wb")
        pickle.dump(solvability_data, f)
        f.close()
    except IOError as err:
        f = open(constants.SOLVABILITYPATH + lvl, "wb")
        pickle.dump([[moves, is_solvable, num_steps]], f)
        f.close()
        


def solvability_with_true_path(lvl):
    """
    Use this to initialize the solvability data
    for a level. It creates a file and populates it
    with prefixes of the move sequence representing
    the true path, which are known to be solvable.
    """
    
    directory = os.getcwd() + "/" + constants.SOLVABILITYPATH
    if not os.path.exists(directory):
        print(directory)
        os.mkdir(directory)
    
    # Get the level and the true solution
    f = open(constants.SOLVEDPATH + lvl, "rb")
    lvl_data = pickle.load(f)
    f.close()
    
    position = utils.PushPosition(lvl_data[0])
    steps = lvl_data[2]
    tot_steps = len(steps)
    
    # Make the steps corresponding to the true position.
    # The PushPosition will translate these steps to moves.
    for step in steps:
        position.step_in_direction(step)
    
    moves = position.moves
    
    print(lvl)
    position = utils.PushPosition(lvl_data[0])
    for (i, move) in enumerate(moves):
        add_data_point(lvl, moves[:i], 1, tot_steps - position.steps)
        print(moves[:i])
        print(tot_steps - position.steps)
        position.make_move(move[0], move[1], move[2])
        
        
def validate_unsolved(lvl, num_rollouts, max_steps, model):
    f = open(constants.SOLVABILITYPATH + lvl, "rb")
    solvability_data = pickle.load(f)
    f.close()
    
    for data_point in solvability_data:
        if data_point[1]:
            continue
        f = open(constants.SOLVEDPATH + lvl, "rb")
        lvl_data = pickle.load(f)
        f.close()
        position = utils.PushPosition(copy.deepcopy(lvl_data[0]))
        
        for move in data_point[0]:
            position.make_move(move[0], move[1], move[2])
            
        result = montecarlo.monte_carlo(position, model, num_rollouts,
                                      512, max_steps, verbosity = 0)
        solvable = result[0] > 0
        if solvable:
            print(position.prettystring())
            print("Found a solution")
            num_steps = result[3]    
            #add_data_point(lvl, position[0], solvable, num_steps)
        else:
            #print(position.prettystring())
            print("Verified unsolved")
        
        
        
def generate_data(lvl, num_rollouts, max_steps, model):
    f = open(constants.SOLVEDPATH + lvl, "rb")
    lvl_data = pickle.load(f)
    f.close()
    
    # First, get the moves for the true path.
    position = utils.PushPosition(copy.deepcopy(lvl_data[0]))
    steps = lvl_data[2]
    
    for step in steps:
        position.step_in_direction(step)
    
    moves = position.moves
    # Now go back to the start and make some of
    # those moves, but not all of them.
    number_moves_to_make = random.randint(0, len(moves) - 2)
    position = utils.get_position(copy.deepcopy(lvl_data[0]),
                                  moves[:number_moves_to_make])
    
    # Make some moves in this position with probability
    # according to the network.
    number_additional_moves = random.randint(2, 10)
    for i in range(number_additional_moves):
        results_var = [0, 0, 0, 1000]
        montecarlo.monte_carlo_advance([position], model, [0], [0], 1,
                                       results_var, 1, 1000)
        if results_var[0] > 0:
            print("Accidentally solved it")
            generate_data(lvl, num_rollouts, max_steps, model)
            return
            
    

    print(position.prettystring())
    moves_for_data = position.moves        
    # Figure out whether this position is possible.
    result = montecarlo.monte_carlo(position, model, num_rollouts,
                                      512, max_steps, verbosity = 0)
    solvable = result[0] > 0
    num_steps = None
    if solvable:
        num_steps = result[3]
    
    print(solvable)
    print(num_steps)
    add_data_point(lvl, moves_for_data, solvable, num_steps)

directory = os.getcwd()+constants.SOLVABILITYPATH
if not os.path.exists(directory):    
    os.mkdir(directory)
    
for lvl in constants.TEST_LEVELS:
    f = open(constants.SOLVABILITYPATH + lvl, "rb")
    solvability_data = pickle.load(f)
    f.close()
    print(len(solvability_data))
    new_s_data = []
    for d in solvability_data:
        new_s_data.append([d[0], d[1], d[2]])
    f = open(constants.SOLVABILITYPATH + lvl, "wb")
    pickle.dump(new_s_data, f)
    f.close()
    print(len(new_s_data))
        
    
#for lvl in constants.TRAIN_LEVELS:
#    solvability_with_true_path(lvl)
 
model = utils.get_model("deep_moveinfo")

for lvl in constants.TEST_LEVELS:
    while True:
        f = open(constants.SOLVABILITYPATH + lvl, "rb")
        solvability_data = pickle.load(f)
        f.close()
        if len(solvability_data) >= 100:
            break
        generate_data(lvl, 64, 200, model)
    

#for datagen in range(1):
#    for lvl in constants.TEST_LEVELS:
#        generate_data(lvl, 64, 200, model)
        
#for lvl in constants.TRAIN_LEVELS:
#    print(lvl)
#    validate_unsolved(lvl, 64, 200, model)