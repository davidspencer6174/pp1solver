import constants
import copy
import utils
import numpy as np

class Node:
    
    def __init__(self, parent, pos, model, is_win = False):
        self.parent = parent
        self.pos = pos
        self.children = dict()
        self.visit_counts = np.zeros(4*constants.SIZE*constants.SIZE)
        self.q = np.zeros(4*constants.SIZE*constants.SIZE)
        self.model = model
        self.is_win = is_win
        self.p = None
        self.v = None
        self.expanded = False
        
    def expand(self):
        self.expanded = True
        most_recent_action = None
        if self.parent is not None:
            most_recent_action = utils.encode(self.pos.moves[-1][0],
                                              self.pos.moves[-1][1],
                                              self.pos.moves[-1][2])
        if self.is_win:
            self.parent.backprop(0, most_recent_action)
            return
        model_result = self.model.predict(np.expand_dims(self.pos.arr, axis=0))
        self.p = model_result[0][0]
        self.v = model_result[1][0][0]
        for move in self.pos.get_legal_move_numbers():
            new_pos = utils.PushPosition(copy.deepcopy(self.pos.arr))
            new_pos.steps = self.pos.steps
            new_pos.moves = copy.deepcopy(self.pos.moves)
            result = new_pos.make_move_number(move)
            self.children[move] = Node(self,
                                       new_pos,
                                       self.model,
                                       result == constants.WIN)
        if self.parent is not None:
            self.parent.backprop(self.v,
                                 most_recent_action)
        return
    
    def backprop(self, estimated_total, a):
        #a is the next action
        if a not in self.children.keys():
            print('bad')
        self.q[a] = (self.q[a]*self.visit_counts[a] + estimated_total)/(self.visit_counts[a] + 1)
        self.visit_counts[a] = self.visit_counts[a] + 1
        if self.parent is not None:
            most_recent_action = utils.encode(self.pos.moves[-1][0],
                                              self.pos.moves[-1][1],
                                              self.pos.moves[-1][2])
            self.parent.backprop(self.pos.steps + estimated_total,
                                 most_recent_action)
    
    def add_to_train_data(self, training_data, total_steps_required):
        training_data[0].append(self.pos.arr)
        training_data[1].append(self.visit_counts/self.visit_counts.sum())
        training_data[2].append((total_steps_required - self.pos.steps)/
                                constants.MAX_STEPS)
        if self.parent is not None:
            self.parent.add_to_train_data(training_data, total_steps_required)

def simulate(current_node, model):
    while current_node.expanded:
        if current_node.is_win:
            most_recent_action = utils.encode(current_node.pos.moves[-1][0],
                                              current_node.pos.moves[-1][1],
                                              current_node.pos.moves[-1][2])
            current_node.parent.backprop(0, most_recent_action)
            return
        if len(current_node.children) == 0:
            current_node.backprop(constants.MAX_STEPS, 0)
            return
        tot_visits = current_node.visit_counts.sum()
        action_goodness = current_node.q +\
                          constants.CPUT*(np.sqrt(1+tot_visits))/\
                          (1+current_node.visit_counts)*current_node.p
        best_action = None
        best_action_value = -1000000
        for act in current_node.children:
            if action_goodness[act] > best_action_value:
                best_action_value = action_goodness[act]
                best_action = act
        current_node = current_node.children[best_action]
    current_node.expand()
    return

def mcts(model, position_orig, max_visits, training_data):
    position = utils.PushPosition(copy.deepcopy(position_orig.arr))
    start_node = Node(None, position, model)
    current_node = start_node
    
    while (current_node.pos.steps < constants.MAX_STEPS and
           current_node.pos.arr[:,:,6:10].sum() != 0):
        while current_node.visit_counts.sum() < max_visits:
            simulate(current_node, model)
        best_action = np.argmax(current_node.visit_counts)
        #print(current_node.visit_counts[best_action])
        #print(current_node.children)
        current_node = current_node.children[best_action]
        if current_node.is_win:
            current_node.parent.add_to_train_data(training_data,
                                           current_node.pos.steps)
            #print('MCTS won!')
            return 1
            break
        #print('took an action {0}'.format(current_node.pos.steps))
    return 0