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
        model_result = self.model.fit(self.pos.arr)
        self.p = model_result[0]
        self.v = model_result[1]
        for move in self.pos.get_legal_move_numbers():
            new_pos = utils.PushPosition(copy.deepcopy(self.pos.arr))
            result = new_pos.make_move_number(move)
            self.children[move] = Node(self,
                                       new_pos,
                                       self.model,
                                       result == constants.WIN)
        if self.parent is not None:
            most_recent_action = utils.encode(self.pos.moves[-1])
            self.parent.backprop(self.pos.steps + self.v*constants.MAX_STEPS,
                                 most_recent_action)
        return
    
    def backprop(self, estimated_total, a):
        #a is the next action
        self.q[a] = (self.q[a]*self.visit_counts[a] + self.v)/(self.visit_counts[a] + 1)
        self.visit_counts[a] = self.visit_counts[a] + 1
        if self.parent is not None:
            most_recent_action = utils.encode(self.pos.moves[-1])
            self.parent.backprop(self.pos.steps + self.v*constants.MAX_STEPS,
                                 most_recent_action)
    
    def add_to_train_data(self, training_data, total_steps_required):
        training_data[0].append(self.pos.arr)
        training_data[1].append(self.visit_counts/self.visit_counts.sum())
        training_data[2].append((total_steps_required - self.pos.steps)/
                                constants.MAX_STEPS)
        if self.parent is not None:
            self.parent.add_to_train_data(training_data, total_steps_required)

def simulate(current_node, model):
    starting_node = current_node
    while current_node.expanded:
        tot_visits = current_node.visit_counts.sum()
        best_action = np.argmax(current_node.q +
                                constants.CPUT*(np.sqrt(1+tot_visits))/
                                (1+current_node.visit_counts)*current_node.p)
        current_node = current_node.children[best_action]
    current_node.expand()
    return

def mcts(model, position_orig, max_visits, training_data):
    position = utils.PushPosition(copy.deepcopy(position_orig.arr))
    success = 0
    start_node = Node(None, position, model)
    current_node = start_node
    won = False
    
    while (position.steps < constants.MAX_STEPS and
           position.arr[:,:,6:10].sum() != 0 and
           not won):
        while current_node.visit_counts.sum() < max_visits:
            simulate(current_node, model)
        best_action = np.argmax(current_node.visit_counts)
        current_node = current_node.children[best_action]
        if current_node.is_win:
            current_node.parent.add_to_train_data(training_data,
                                           current_node.pos.steps)
    return (success, ([], [], []))