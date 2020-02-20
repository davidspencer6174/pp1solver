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

netname = "deep_moveinfo"

model = None
train = False
if train:
    initializing = True
    model = perform_training(initializing, netname, numlayers = 12,
                             epochs = 3, training_sets = 10, batch_size = 64,
                             learning_rate = .0002)
else:
    model = get_model(netname)

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
#"9by8", "Crazystylie", "HouseofGod", "Tetris", "k2xlgames",
#            "AllBoxedUp", "TrickQuestion", "Corner2Corner"
for lvl in ["9by8",
            "AllBoxedUp", "TrickQuestion", "Corner2Corner"]:#,
#for lvl in ["Crumpet"]:
            #"AllBoxedUp", "TrickQuestion", "Corner2Corner"]:
    
    print("tree searching for " + lvl)
    init_position = utils.import_raw_level(lvl, rawpath)[0]
    tree_search(init_position, model, 40000, verbosity = 1)
    print("monte carlo searching for " + lvl)
    monte_carlo(init_position, model, 8192, 512, 100, verbosity = 2)