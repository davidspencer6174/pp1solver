import constants
import copy
import utils
import numpy as np
import gc
import tensorflow.keras
import predictionbatcher


class Node:
    
    def __init__(self, parent, moves, model, pred_batcher, transposition_table,
                 proximity_bonus = False, is_win = False):
        self.parent = parent
        self.still_updating = True
        self.moves = moves
        self.children = dict()
        self.transposition_table = transposition_table
        num_classes = 4
        self.visit_counts = np.zeros(num_classes)
        self.q = np.zeros(num_classes)
        self.q += 20
        #if self.parent is not None:
        #    self.q -= (self.parent.v-1)*constants.MAX_STEPS/2
        self.legal_moves = None
        self.move_results = None
        self.model = model
        self.proximity_bonus = proximity_bonus
        self.is_win = is_win
        self.p = None
        self.v = None
        self.pred_batcher = pred_batcher
        self.expanded = False
        
    def expand(self, pos):
        for move in self.moves:
            res = pos.make_move_number(move)
        pos_hash = pos.get_hash()
        #if pos_hash in self.transposition_table:
        #    other_node = self.transposition_table[pos_hash]
        #    move_diff = len(self.moves) - len(other_node.moves)
        self.transposition_table[pos.get_hash()] = self
        self.expanded = True
        most_recent_action = None
        if self.parent is not None:
            most_recent_action = self.moves[-1]
        if self.is_win:
            self.parent.backprop(min(pos.steps, constants.MAX_STEPS),
                                 most_recent_action)
            return
        #self.legal_moves = pos.get_legal_move_numbers()
        self.legal_moves = []
        self.move_results = []
        for direction in range(4):
            result = pos.make_move_number(direction, actually_move = False)
            self.move_results.append(result)
            if result != constants.ILLEGAL:
                self.legal_moves.append(direction)
        if len(self.legal_moves) == 0 or len(self.moves) > constants.MAX_STEPS:
            self.parent.backprop(constants.MAX_STEPS + 20, most_recent_action)
            return
        self.pred_batcher.add_inp(pos.arr, self)
        
    def expand_result_callback(self, p, v):
        self.p = p
        self.v = v
        for move in self.legal_moves:
            new_moves = copy.deepcopy(self.moves) + [move]
            result = self.move_results[move]
            self.children[move] = Node(self,
                                       new_moves,
                                       self.model,
                                       self.pred_batcher,
                                       self.transposition_table,
                                       proximity_bonus = self.proximity_bonus,
                                       is_win = (result == constants.WIN))
        if self.parent is not None:
            most_recent_action = self.moves[-1]
            self.parent.backprop((self.v+1)*constants.MAX_STEPS/2 + len(self.moves),
                                 most_recent_action)
        return
    
    def backprop(self, estimated_total, a):
        #a is the next action
        if not self.still_updating:
            return
        if a not in self.children.keys():
            print('bad')
            print(self.is_win)
            print(self.moves)
            print(a)
            print(self.children.keys())
            #print(self.pos.visited_since_push)
        self.q[a] = (self.q[a]*self.visit_counts[a] + constants.MAX_STEPS - estimated_total)/(self.visit_counts[a] + 1)
        self.visit_counts[a] = self.visit_counts[a] + 1
        if self.parent is not None:
            most_recent_action = self.moves[-1]
            self.parent.backprop(estimated_total,
                                 most_recent_action)
    
    def add_to_train_data(self, training_data, remaining_moves,
                          total_steps_required, was_win, pos):
        #print('adding data')
        if len(remaining_moves) == 0:
            return
        steps_for_net = total_steps_required - pos.steps
        if not was_win:
            steps_for_net = constants.MAX_STEPS
        training_data[0].append(copy.copy(pos.arr))
        training_data[1].append(self.visit_counts/self.visit_counts.sum())
        #note: low means good
        training_data[2].append(steps_for_net/constants.MAX_STEPS*2 - 1)
        pos.make_move_number(remaining_moves[0])
        self.children[remaining_moves[0]].add_to_train_data(training_data,
                                                            remaining_moves[1:],
                                                            total_steps_required,
                                                            was_win,
                                                            pos)


def simulate(current_node, model, orig_pos):
    while current_node.expanded:
        most_recent_action = None
        if len(current_node.moves) > 0:
            most_recent_action = current_node.moves[-1]
        if current_node.is_win:
            current_node.parent.backprop(min(len(current_node.moves),
                                             constants.MAX_STEPS),
                                         most_recent_action)
            return
        if len(current_node.children) == 0:
            current_node.parent.backprop(constants.MAX_STEPS+20, most_recent_action)
            return
        tot_visits = current_node.visit_counts.sum()
        action_goodness = current_node.q/constants.MAX_STEPS +\
                          constants.CPUT*(np.sqrt(1+tot_visits))/\
                          (1+current_node.visit_counts)*current_node.p
        #if max(current_node.visit_counts) > 125 and len(current_node.children) > 1:
        #    print(action_goodness)
        #    print(current_node.visit_counts)
        #    print(current_node.p)
        #    print(current_node.q)
        #    print(tot_visits)
        #    print(orig_pos.prettystring())
        best_action = None
        best_action_value = -1000000
        for act in current_node.children:
            if current_node.proximity_bonus:
                if act == 0 and current_node.pos.char_loc[0] > current_node.pos.exit_loc[0]:
                    action_goodness[act] += .1
                if act == 1 and current_node.pos.char_loc[1] < current_node.pos.exit_loc[1]:
                    action_goodness[act] += .1
                if act == 2 and current_node.pos.char_loc[0] < current_node.pos.exit_loc[0]:
                    action_goodness[act] += .1
                if act == 3 and current_node.pos.char_loc[1] > current_node.pos.exit_loc[1]:
                    action_goodness[act] += .1
            if action_goodness[act] > best_action_value:
                best_action_value = action_goodness[act]
                best_action = act
        current_node = current_node.children[best_action]
    orig_position_copy = utils.StepPosition(copy.deepcopy(orig_pos.arr))
    current_node.expand(orig_position_copy)
    return

def mcts_multipos(model, positions, num_visits, training_data, overall_iter):
    current_nodes = []
    orig_nodes = []
    #proximity_bonus = (overall_iter < 7)
    proximity_bonus = False
    pred_batcher = predictionbatcher.PredictionBatcher(model)
    done = np.zeros(len(positions))
    succeeded = np.zeros(len(positions))
    for pos in positions:
        current_nodes.append(Node(None, [], model, pred_batcher,
                                  dict(),
                                  proximity_bonus))
        orig_nodes.append(current_nodes[-1])
    while done.min() == 0:        
        it = 0
        while min([current_nodes[i].visit_counts.sum()
                   for i in range(len(positions)) if done[i] == 0]) < num_visits:
            it += 1
            if it > num_visits*2:
                print('uh')
        #for visit in range(max_visits):
            for i in range(len(positions)):
                if done[i] == 0:
                    simulate(current_nodes[i], model, positions[i])
            pred_batcher.predict()
        for i in range(len(positions)):
            if done[i] == 1:
                continue
            best_action = np.argmax(current_nodes[i].visit_counts)
            current_nodes[i].still_updating = False
            current_nodes[i] = current_nodes[i].children[best_action]
            if current_nodes[i].is_win:
                orig_position_copy = utils.StepPosition(copy.deepcopy(positions[i].arr))
                orig_nodes[i].add_to_train_data(training_data,
                                                current_nodes[i].moves,
                                                len(current_nodes[i].moves),
                                                True,
                                                orig_position_copy
                                                )
                done[i] = 1
                succeeded[i] = 1
                continue
            if len(current_nodes[i].moves) > constants.MAX_STEPS or\
            len(current_nodes[i].legal_moves) == 0:
                #Not sure whether we should do this: include
                #failures in train data
                orig_position_copy = utils.StepPosition(copy.deepcopy(positions[i].arr))
                orig_nodes[i].add_to_train_data(training_data,
                                                current_nodes[i].moves,
                                                constants.MAX_STEPS,
                                                False,
                                                orig_position_copy
                                                )
                done[i] = 1
                continue
            #if not current_nodes[i].expanded:
            #    print('interesting')
            #if current_nodes[i].legal_moves is None:
            #    print('interesting2')
            #if current_nodes[i].is_win:
            #    print('interesting2a')
            #if len(current_nodes[i].legal_moves) == 0:
            #    print('interesting3')
        print('{0} done, {1} succeeded'.format(done.sum(), succeeded.sum()))
    return succeeded