import pickle
import copy
import random

import tensorflow as tf
from tensorflow.keras.models import model_from_json

import utils
import numpy as np
import constants
import model_training
import mcts

level_path = constants.RAWPATH
level_name = 'Sandstorm'
start_pos = utils.import_raw_level(level_name)[0]
training_data = ([], [], [])

model = utils.initialize_model(6, .001)
#num_playouts = 1600
num_playouts = 40
winrate_threshold = .8

for proportion in np.linspace(.00, 1.0, 21):
    while True:
        print(f'Made it to {proportion}')
        num_curriculum_levels = 30
        successes = 0
        for i in range(num_curriculum_levels):
            curriculum_pos = utils.make_curriculum_pos(start_pos, proportion)
            success = mcts.mcts(model,
                                curriculum_pos,
                                num_playouts,
                                training_data)
            successes += success
            print(success)
            #if success:
            #    for tup in range(3):
            #        training_data[tup] += new_training_data[tup]
        training_data = training_data[-10000:]
        print(successes/num_curriculum_levels)
        if successes/num_curriculum_levels > winrate_threshold:
            break
        model_training.train_model(model, training_data)
        