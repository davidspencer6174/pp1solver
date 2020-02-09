import copy
import numpy as np
import utils

def monte_carlo(init_position, model, num_simulations, batch_size, move_limit):
    orig_arr = copy.deepcopy(init_position.arr)
    #Positions are stored as move sequences rather than
    #as full arrays, which are less memory efficient.
    positions = [utils.PushPosition(copy.deepcopy(orig_arr)) for i in range(num_simulations)]
    terminated = np.zeros(num_simulations)
    solved = 0
    nolegal = 0
    outofmoves = 0
    best_soln = 1000
    while 0 in terminated:
        nonterminated_inds = np.where(terminated == 0)[0]
        num_left = len(nonterminated_inds)
        #print("Starting iteration. " + str(num_left) + " left.")
        querying = np.zeros((num_left, 20, 20, 12))
        for i in range(num_left):
            querying[i,:,:,:] = copy.deepcopy(positions[nonterminated_inds[i]].arr)
            utils.position_transform(querying[i,:,:,:])
        predictions = model.predict(querying, batch_size = batch_size)
        for i in range(num_left):
            move_to_make = np.random.choice(20*20*4, p=predictions[i,:]/np.sum(predictions[i,:]))
            pos = positions[nonterminated_inds[i]]
            move_result = pos.make_move_number(move_to_make)
            while move_result == -1:
                #print("Illegal move")
                #terminated[nonterminated_inds[i]] = 1
                predictions[i,move_to_make] = 0
                if np.max(predictions[i,:]) == 0:
                    nolegal += 1
                    terminated[nonterminated_inds[i]] = 1
                    break
                move_to_make = np.random.choice(20*20*4, p=predictions[i,:]/np.sum(predictions[i,:]))
                pos = positions[nonterminated_inds[i]]
                move_result = pos.make_move_number(move_to_make)
                #if move_result != -1:
                #    print("Failed to have legal move on top")
            if move_result == 10000:
                solved += 1
                if -pos.moves_penalty < best_soln:
                    print("New best solution: " + str(-pos.moves_penalty))
                    best_soln = -pos.moves_penalty
                terminated[nonterminated_inds[i]] = 1
            if -pos.moves_penalty >= move_limit:
                outofmoves += 1
                terminated[nonterminated_inds[i]] = 1
    print("Number that solved: " + str(solved))
    print("Number that ran out of legal moves: " + str(nolegal))
    print("Number that ran out of moves: " + str(outofmoves))
    print("Best solution found: " + str(best_soln))
                
    
    