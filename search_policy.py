import copy
import tensorflow as tf
import utils
from tensorflow.keras.models import model_from_json
import numpy as np
from heapq import heappush, heappop
from itertools import count
from time import clock
import constants

# A first attempt at a search. Each node in the tree represents a
# position attained by some sequence of moves from the starting position.
# The search "expands" nodes (that is, creates nodes for all possible
# moves from the position represented by the node) until it finds
# winning nodes, and then starts terminating other branches
# based on Manhattan distance.


# This is not a sophisticated search. The only information used
# to determine which nodes to expand next is the number of
# steps, the history of the policy values at each move, and
# the history of the number of legal moves at each move.

# We hope to add some way of telling which positions are more
# or less likely to be solvable.

def tree_search(full_init_position, model, positions_to_check,
                batch_size = 128, verbosity = 1):
    """
    full_init_position: a PushPosition for the starting position
    to search from
    model: a neural network which outputs policy values
    positons_to_check: the number of positions to expand during
    the search
    batch_size: the number of positions to process at once
    verbosity: 0, 1, or 2, determines how many debug messages
    are printed
    """
    # Have a tiebreaker so that the heapq always has some way of
    # determining which node should be expanded next.
    tiebreaker = count()
    
    # Make a copy of the original state.
    original_arr = copy.deepcopy(full_init_position.arr)
    original_char_loc = copy.deepcopy(full_init_position.char_loc)
    size = full_init_position.size
    
    # The first member in the tuple is the priority.
    # The second member is a tiebreaker.
    # The third member represents the position.
    # The fourth member represents the number of steps taken.
    seed_position = (0, next(tiebreaker), [], 0)
    
    positions = []  # This will be used as a heapq of positions.
    # For the purpose of memory efficiency, the positions are
    # stored as a list of moves rather than as a position array.
    # Whenever we need an array to pass to the net, we return to
    # the original position and use the moves to reconstruct the
    # current position. This costs time but saves memory.
    
    # Push the first position onto the heap with priority 0.
    heappush(positions, seed_position)
    
    # Variables to track the status of the search
    deepest_explored = 0
    positions_checked = 0
    
    # For now, assume a limit on solution length (will need to be
    # increased for many levels).  Of course, it would be ideal
    # to eventually do away with this.
    shortest_soln_length = 100
    shortest_soln = []
    all_solutions = []

    start_time = clock()
    end_time = clock()
    
    while positions_checked < positions_to_check:
        if verbosity > 1:
            print("Positions expanded so far: {}".format(positions_checked))
            print("Deepest branch so far: {}".format(deepest_explored))
            print("Best solution length so far: {}".format(shortest_soln_length))
            print("Number of positions in queue: {}".format(len(positions)))
        # If all branches have been terminated, the search is over.
        if len(positions) == 0:
            break
        # Prepare to store the positions and corresponding
        # priorities for the nodes we will expand.
        old_positions = []
        querying = []
        old_x_priorities = []
        # This loop takes the top batch_size positions by
        # priority and prepares them to be sent through the net.
        for i in range(min(batch_size, len(positions))):
            # Get the highest-priority position (encoded as a
            # sequence of moves)
            position = heappop(positions)
            if i == 0 and verbosity > 1:
                print("Current penalty: {}".format(position[0]))
            # Get from the original position to the high-priority position
            #old_positions.append(utils.get_position(copy.deepcopy(original_arr), 
            #                                        position[2]))
            old_positions.append(utils.get_position_faster(copy.deepcopy(original_arr),
                                                           position[2],
                                                           position[3]))
            # Add the array to the net input
            querying.append(copy.deepcopy(old_positions[-1].arr))
            # Perform the transformation required to pass the
            # position into the neural network
            utils.position_transform(querying[-1])
            # Remember the priority for the parent position
            old_x_priorities.append(position[0])
            positions_checked += 1
        querying = np.array(querying)
        # Pass through the net
        predictions = model.predict(querying)
        for i in range(len(old_positions)):
            # Escape if finding a better solution is hopeless
            exit_loc = old_positions[i].exit_loc
            char_loc = old_positions[i].char_loc
            manhat = abs(exit_loc[0]-char_loc[0]) + abs(exit_loc[1]-char_loc[1])
            if old_positions[i].steps + manhat > shortest_soln_length:
                continue
            # Update depth statistic (for user to see)
            if old_positions[i].steps > deepest_explored:
                deepest_explored = old_positions[i].steps
            # Count the legal moves (to contextualize probabilities)
            num_legal = len(np.nonzero(old_positions[i].arr[:,:,6:10]))
            position_copy = copy.deepcopy(old_positions[i])
            old_steps = position_copy.steps
            # Now try each possible move and add to the positions queue
            max_move = size*size*4
            for move in range(0, max_move):
                move_result = position_copy.make_move_number(move)
                if move_result == constants.ILLEGAL:
                    continue
                elif move_result == constants.WIN:  # Found a win
                    end_time = clock()
                    if position_copy.steps >= shortest_soln_length:
                        continue
                    shortest_soln_length = position_copy.steps
                    if verbosity > 0:
                        print("Found win of length {}".format(shortest_soln_length))
                    shortest_soln = position_copy.moves
                    position_copy = copy.deepcopy(old_positions[i])
                    # Record the discovery of this solution
                    all_solutions.append((position_copy.moves,
                                          clock()-start_time,
                                          shortest_soln_length))
                else:
                    exit_loc = position_copy.exit_loc
                    char_loc = position_copy.char_loc
                    manhat = abs(exit_loc[0]-char_loc[0]) + abs(exit_loc[1]-char_loc[1])
                    if position_copy.steps + manhat > shortest_soln_length:
                        position_copy = copy.deepcopy(old_positions[i])
                        continue
                    move_diff = - old_steps + position_copy.steps
                    # This formula is unfortunately full of magic numbers.
                    # The idea is to slightly penalize long paths and reward
                    # paths with lots of moves that are highly likely
                    # relative to the number of legal moves.
                    # I am not sure of the most principled way to do this.
                    new_priority = (old_x_priorities[i]
                                    + .05*move_diff
                                    - np.log(predictions[i][move]*num_legal**.75))
                    #print(new_priority)
                    #print(position_copy.moves)
                    heappush(positions, (new_priority, next(tiebreaker),
                                         position_copy.moves, position_copy.steps))
                    position_copy = copy.deepcopy(old_positions[i])
    if verbosity > 0:
        print("Shortest solution found has  {} moves".format(shortest_soln_length))
        print("Solution: {}".format(shortest_soln))
        print("Time to find solution: {} seconds".format(end_time-start_time))
        print("All solutions along with times:")
        for i in all_solutions:
            print(i)