import copy
import tensorflow as tf
import utils
from keras.models import model_from_json
import numpy as np
from heapq import heappush, heappop
from time import clock

# A first attempt at a search.
# This is not a sophisticated search. It expands nodes based
# on a simple 

rawpath = "RawLevels/"
netpath = "networks/policy.json"
weights = "networks/policy.h5"

#level_name = "9by8"
#level_name = "LOLevel"
#level_name = "RabbitHole"
level_name = "K2xlgames"

full_init_position = utils.import_raw_level(level_name, rawpath)[0]
original_arr = copy.deepcopy(full_init_position.arr)
original_char_loc = copy.deepcopy(full_init_position.char_loc)
size = full_init_position.size

# The first member in the tuple is the priority.
# The second member represents the position.
seed_position = (0, [])

positions = []  # This will be used as a heapq of positions.
# For the purpose of memory efficiency, the positions are
# stored as a list of moves rather than as a position array.
# For passing through the net, one can return to the original
# position and use the moves to reconstruct the current position.

heappush(positions, seed_position)

json_file = open(netpath, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(weights)

positions_to_check = 100000
deepest_explored = 0
positions_checked = 0

# For now, assume a limit on solution length (will need to be
# increased for many levels).  Of course, it would be ideal
# to eventually do away with this.
shortest_soln_length = 100
shortest_soln = []
all_solutions = []

batch_size = 32
start_time = clock()
end_time = clock()

while positions_checked < positions_to_check:
    print("Positions expanded so far: {}".format(positions_checked))
    print("Deepest branch so far: {}".format(deepest_explored))
    print("Best solution length so far: {}".format(shortest_soln_length))
    print("Number of positions in queue: {}".format(len(positions)))
    if len(positions) == 0:
        break
    old_positions = []
    querying = []
    old_x_priorities = []
    # The first loops takes the top batch_size positions by
    # priority and prepares them to be sent through the net.
    for i in range(min(batch_size, len(positions))):
        position = heappop(positions)
        if i == 0:  # Just for tracking progress
            print("Current penalty: {}".format(position[0]))
        # Now get from the original position to the current position
        old_positions.append(utils.get_position(copy.deepcopy(original_arr), 
                                                position[1]))
        # Center the character and add the array to the net input
        querying.append(utils.centered(old_positions[-1].arr,
                                       old_positions[-1].char_loc[0],
                                       old_positions[-1].char_loc[1]))
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
        if -old_positions[i].moves_penalty + manhat > shortest_soln_length:
            continue
        # Update depth statistic (for user to see)
        if -old_positions[i].moves_penalty > deepest_explored:
            deepest_explored = -old_positions[i].moves_penalty
        # Count the legal moves (to contextualize probabilities)
        num_legal = len(np.nonzero(old_positions[i].arr[:,:,6:10]))
        position_copy = copy.deepcopy(old_positions[i])
        oldmoves_penalty = position_copy.moves_penalty
        # Now try each possible move and add to the positions queue
        for move in range(0, (2*size-1)*(2*size-1)*4):
            move_result = position_copy.make_move_number(move)
            if move_result == -1:
                continue
            elif move_result == 10000:  # Found a win
                end_time = clock()
                if -position_copy.moves_penalty >= shortest_soln_length:
                    continue
                shortest_soln_length = -position_copy.moves_penalty
                print("Found win of length {}".format(shortest_soln_length))
                shortest_soln = position_copy.moves
                position_copy = copy.deepcopy(old_positions[i])
                # Record the discovery of this solution
                all_solutions.append((position_copy.moves,
                                      clock()-start_time,
                                      shortest_soln_length))
            else:
                move_diff = oldmoves_penalty - position_copy.moves_penalty
                # This formula is unfortunately full of magic numbers.
                # The idea is to slightly penalize long paths and reward
                # paths with lots of moves that are highly likely
                # relative to the number of legal moves.
                # I am not sure of the most principled way to do this.
                new_priority = (old_x_priorities[i]
                                + .05*move_diff
                                - np.log(predictions[i][move]*num_legal**.75))
                heappush(positions, (new_priority, position_copy.moves))
                position_copy = copy.deepcopy(old_positions[i])
                
print("Shortest solution found has  {} moves".format(shortest_soln_length))
print("Solution: {}".format(shortest_soln))
print("Time to find solution: {} seconds".format(end_time-start_time))
print("All solutions along with times:")
for i in all_solutions:
    print(i)