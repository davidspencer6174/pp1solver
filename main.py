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

rawpath = "RawLevels/"
netname = "deep_moveinfo"

model = None
train = False
if train:
    initializing = False
    model = perform_training(initializing, netname, numlayers = 12,
                             epochs = 5, training_sets = 10, batch_size = 512)
else:
    netpath = "networks/policy_" + str(netname) + ".json"
    weights = "networks/policy_" + str(netname) + ".h5"
    json_file = open(netpath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)

#"9by8", "Crazystylie", "HouseofGod", "Tetris", "k2xlgames",
#            "AllBoxedUp", "TrickQuestion", "Corner2Corner"
for lvl in ["Crazystylie"]:#,
            #"AllBoxedUp", "TrickQuestion", "Corner2Corner"]:
    
    print("tree searching for " + lvl)
    init_position = utils.import_raw_level(lvl, rawpath)[0]
    tree_search(init_position, model, 10000, verbosity = 2)
    #print("running monte carlo for " + lvl)
    #init_position = utils.import_raw_level(lvl, rawpath)[0]
    #monte_carlo(init_position, model, 8192, 32, 100, verbosity = 2)