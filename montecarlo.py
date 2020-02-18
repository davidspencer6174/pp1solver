import copy
import numpy as np
import utils
import constants

def monte_carlo(init_position, model, num_simulations, batch_size, move_limit,
                verbosity = 1):
    """
    Performs in parallel num_simulations Monte Carlo rollouts.
    Uses model as the neural network to determine the probability
    to take each possible move.
    Each simulation terminates once one of the following conditions
    occurs:
        1. A win is reached
        2. No legal moves with nonzero probability according
        to the net exist in the position
        3. The move_limit is reached. This move limit is
        updated whenever a solution is found to prevent
        extra computational work finding slower solutions.
    """
    orig_arr = copy.deepcopy(init_position.arr)
    # Since the number of positions here is not too large,
    # we store them as full positions rather than move sequences
    # as in the tree search.
    # If we want to do larger simulations, we could break them
    # up into several simulations of, say, 2^16 each.
    
    # Make many copies of the position, one per simulation
    positions = [utils.PushPosition(copy.deepcopy(orig_arr)) for i in range(num_simulations)]
    # Track which simulations have terminated
    terminated = np.zeros(num_simulations)
    # Keep summary statistics of how many simulations
    # terminate each way
    solved = 0
    nolegal = 0
    outofmoves = 0
    
    best_soln = 1000
    
    # While loop until every simulation has terminated
    while 0 in terminated:
        # Figure out which simulations have not terminated and
        # how many of them there are
        nonterminated_inds = np.where(terminated == 0)[0]
        num_left = len(nonterminated_inds)
        if verbosity > 1:
            print("Starting iteration. " + str(num_left) + " left.")
        # Put the nonterminated positions in a numpy array
        # to pass into the net
        querying = np.zeros((num_left, 20, 20, 12))
        for i in range(num_left):
            querying[i,:,:,:] = copy.deepcopy(positions[nonterminated_inds[i]].arr)
            utils.position_transform(querying[i,:,:,:])
        predictions = model.predict(querying, batch_size = batch_size)
        for i in range(num_left):
            # Choose a move with probability proportional to
            # the net output.
            move_to_make = np.random.choice(20*20*4, p=predictions[i,:]/np.sum(predictions[i,:]))
            pos = positions[nonterminated_inds[i]]
            move_result = pos.make_move_number(move_to_make)
            # While the move is illegal, zero its probability out
            # and choose another move.
            while move_result == constants.ILLEGAL:
                predictions[i,move_to_make] = 0
                # If no moves are still viewed as possibilities,
                # we have reached a terminating condition.
                if np.max(predictions[i,:]) == 0:
                    nolegal += 1
                    terminated[nonterminated_inds[i]] = 1
                    break
                move_to_make = np.random.choice(20*20*4, p=predictions[i,:]/np.sum(predictions[i,:]))
                pos = positions[nonterminated_inds[i]]
                move_result = pos.make_move_number(move_to_make)
                #if move_result != -1:
                #    print("Failed to have legal move on top")
            # If we have reached a win, terminate the
            # simulation and store the number of moves.
            if move_result == constants.WIN:
                solved += 1
                if -pos.steps < best_soln:
                    print("New best solution: " + str(-pos.steps))
                    best_soln = -pos.steps
                terminated[nonterminated_inds[i]] = 1
            # If we have reached the move limit, terminate
            # the simulation.
            if -pos.steps >= move_limit or -pos.steps > best_soln:
                outofmoves += 1
                terminated[nonterminated_inds[i]] = 1
    if verbosity > 0:
        print("Number that solved: " + str(solved))
        print("Number that ran out of legal moves: " + str(nolegal))
        print("Number that ran out of moves: " + str(outofmoves))
        print("Best solution found: " + str(best_soln))
                
    
    