import pickle
import copy
import random
import os
import gc

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import model_from_json

import utils
import numpy as np
import constants
import model_training
import mcts

level_path = constants.RAWPATH
level_names = ['5by5', '9by8', 'Sandstorm', 'LOLevel', 'WetSalmon', 'K2xlgames',
               'LeakInTheRoof', 'Superlock', 'TrapDoor', 'TinMan', 'LinktoPast',
               'Octopussy', 'CupOfWaffles']
recent_performances = {lvl: [] for lvl in level_names}
proportions = {lvl: .00 for lvl in level_names}
#level_names = ['9by8']
training_data = ([], [], [])

model = None
load_existing = True
if load_existing:
    json_file = open("networks/latest.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("networks/latest.h5")
    model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=constants.LR),
                  loss = [tensorflow.keras.losses.categorical_crossentropy,
                          tensorflow.keras.losses.MSE])
    with open('networks/perfs.pickle', 'rb') as handle:
        recent_performances, proportions = pickle.load(handle)
else:
    model = utils.initialize_model(11)
#num_playouts = 1600
num_playouts = 512
winrate_threshold = .8
poses_to_save = 50000
overall_iter = 0

while min(proportions.values()) < 1.0:
    levels_chosen = []
    print(proportions)
    print(recent_performances)
    training_maxes = np.array([max(q) for q in training_data[1]])
    print('Mean max: {0}'.format(training_maxes.mean()))
    num_curriculum_levels = 100
    curriculum_poses = []
    successes = 0
    for i in range(num_curriculum_levels):
        level_name = random.choice(level_names)
        levels_chosen.append(level_name)
        start_pos = utils.import_raw_level(level_name)[0]
        curriculum_pos = utils.make_curriculum_pos(start_pos, proportions[level_name])
        curriculum_poses.append(curriculum_pos)
        
    successes_arr = mcts.mcts_multipos(model,
                                       curriculum_poses,
                                       num_playouts,
                                       training_data,
                                       overall_iter)
    for i in range(len(successes_arr)):
        recent_performances[levels_chosen[i]].append(successes_arr[i])
    for lvl in level_names:
        recent_performances[lvl] = recent_performances[lvl][-10:]
        if len(recent_performances[lvl]) < 3:
            continue
        avg_perf = sum(recent_performances[lvl])/len(recent_performances[lvl])
        if avg_perf > .7:
            proportions[lvl] = min(proportions[lvl] + .05, 1.0)
            recent_performances[lvl] = []
        
    training_data = (training_data[0][-poses_to_save:],
                     training_data[1][-poses_to_save:],
                     training_data[2][-poses_to_save:])
    if len(training_data[0]) > 0:
        model_training.train_model(model, training_data)
    
    # Save the trained network
    model_json = model.to_json()
    directory = os.getcwd()+'/networks'
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open("networks/latest.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("networks/latest.h5")
    del model
    tensorflow.keras.backend.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()
    #model = utils.initialize_model(6, .001)
    
    json_file = open("networks/latest.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("networks/latest.h5")
    model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=constants.LR),
                  loss = [tensorflow.keras.losses.categorical_crossentropy,
                          tensorflow.keras.losses.MSE])
    with open('networks/perfs.pickle', 'wb') as handle:
        pickle.dump((recent_performances, proportions), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    overall_iter += 1
    