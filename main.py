import copy
import tensorflow as tf
import utils
from tensorflow.keras.models import model_from_json
import numpy as np
from heapq import heappush, heappop
from itertools import count
from montecarlo import monte_carlo
from time import clock
from search_policy import tree_search
from training_supervised_policy import perform_training

"""
Depending on the booleans train and initializing, this file can
do any of the following three things:
    
    1. Train a new net and test it. This happens if train = True and
    initializing = True.
    2. Train an existing net and test it. This happens if train = True
    and initializing = False.
    3. Test an existing net. This happens if train = False.
"""

netname = "deep_nocharinfo"

model = None
train = False
if train:
    initializing = True
    model = perform_training(initializing, netname, numlayers = 12,
                             epochs = 1, training_sets = 10, batch_size = 64,
                             learning_rate = .0002)
else:
    model = utils.get_model(netname)

# List of some test levels. Make sure not to use levels
# that were used for training.
# Optimal stepcounts:
# 9by8: 63
# Crazystylie: 33
# House of God: 41
# Corner2Corner: 58
# Trick Question: 76
# All Boxed Up: 48
# Tetris: 61
# crazystylie: 33
# k2xlgames: 47

#deep_moveinfo: optimally solved k2xlgames, crazystylie
#"9by8", "Crazystylie", "HouseofGod", "Tetris", "k2xlgames",
#            "AllBoxedUp", "TrickQuestion", "Corner2Corner"
for lvl in ["Crazystylie"]:
#for lvl in ["Crumpet"]:
            #"AllBoxedUp", "TrickQuestion", "Corner2Corner"]:
    init_position = utils.import_raw_level(lvl)[0]
    #print("monte carlo searching for " + lvl)
    #monte_carlo(init_position, model, 4096, 512, 100, verbosity = 2)
    print("tree searching for " + lvl)
    tree_search(init_position, model, 250000, verbosity = 2)