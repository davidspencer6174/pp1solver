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
level_name = '9by8'
start_pos = utils.import_raw_level(level_name)[0]
training_data = ([], [], [])

model = utils.initialize_model(6, .001)
#num_playouts = 1600
num_playouts = 128
winrate_threshold = .8
poses_to_save = 3000

for proportion in np.linspace(.50, 1.0, 11):
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
            print(str(success)+' '+str(i))
            #if success:
            #    for tup in range(3):
            #        training_data[tup] += new_training_data[tup]
        training_data = (training_data[0][-poses_to_save:],
                         training_data[1][-poses_to_save:],
                         training_data[2][-poses_to_save:])
        print(successes/num_curriculum_levels)
        model_training.train_model(model, training_data)
        if successes/num_curriculum_levels > winrate_threshold:
            break
        
        # Save the trained network
        model_json = model.to_json()
        directory = os.getcwd()+'/networks'
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open("networks/"+level_name+".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("networks/"+level_name+".h5")
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
        model = utils.initialize_model(6, .001)
        
        json_file = open("networks/"+level_name+".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        