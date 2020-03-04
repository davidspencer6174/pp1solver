#Usage: run python gameplay.py level_name (where level_name is the
#file name of the level) 

#This program allows one to play levels in such a way that
#gameplay data will be saved for training if you choose.
#to use it, go to a PP1-style level on the PP2 website
#and save its level code to your computer.

#PP2 website: k2xl.com/games/psychopath2/?go=index

#gameplay.py assumes that there is a folder called RawLevels in the
#same folder as gameplay.py, which is used to store the level code from
#the website. It also assumes that there is a folder called SolvedLevels
#which is used to store the levels with solutions.

#Play the level and press s when you reach the exit
#if you want it saved (it won't be saved otherwise).
#You can undo by pressing u or restart by pressing r.
#You can change the folder names in the lines after the imports.

import utils as utils
import tensorflow as tf
from tensorflow.keras.models import model_from_json

import numpy as np

import os
#**********************************
#Change the folder names here
rawpath = "RawLevels/"
outpath = "SolvedLevels/"
#**********************************


def draw(width, height, p, steps, probabilities, using_net):    #draw the grid
    screen.fill(white)
    dim = max(width, height)
    sq_size = 350.0/dim
    for i in range(0, height):
        for j in range(0, width):
            locations = [25+sq_size*j, 25+sq_size*i, sq_size, sq_size]
            # Unmovable
            if p.is_unmovable(i, j):
                pg.draw.rect(screen, black, locations)
            # Movable or movable blocking exit
            if p.is_movable(i, j):
                pg.draw.rect(screen, grey, locations)
            # Character
            if p.is_char(i, j):
                pg.draw.rect(screen, yellow, locations)
            # Exit
            if p.is_win(i, j) and not p.is_movable(i, j):
                pg.draw.rect(screen, blue, locations)
            if not using_net:
                continue
            square_probs = probabilities[i*80+j*4:i*80+j*4+4]
            center = np.array([25 + sq_size*(j + 0.5), 25 + sq_size*(i + 0.5)])
            vecs = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
            for direction in range(4):
                if square_probs[direction] > .01:
                    line_end = center + square_probs[direction] * vecs[direction] * sq_size/2
                    pg.draw.line(screen, black, center, line_end, 1)
    # Now draw grid
    for i in range(0, width + 1):
        pg.draw.line(screen, black, [25+sq_size*i, 25],
                     [25+sq_size*i, 25+sq_size*height], 3)
    for i in range(0, height + 1):
        pg.draw.line(screen, black, [25, 25+sq_size*i],
                     [25+sq_size*width, 25+sq_size*i], 3)
    # Display stepcount
    font = pg.font.SysFont("monospace", 15)
    label = font.render(str(len(steps)), 1, (0, 0, 0))
    screen.blit(label, (400, 25))
    pg.display.flip()
    
     
import copy
import pickle
import sys

import pygame as pg


white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255, 255, 0)
grey = (128, 128, 128)
blue = (0, 0, 255)
levels=os.listdir(path='RawLevels')

done = True
print('Which level would you like to play? \n(hit enter for random level) \n(hit 1 for list of levels)')
level_name=str(input())
while done == True:
	if level_name=='1':
		for i in levels:
			print(i[:-4])	
		print('Which level would you like to play? \n(hit enter for random level) \n(hit 1 for list of levels)')
		level_name=str(input())
	elif level_name+'.txt' not in levels:
		level_name=levels[np.random.random_integers(0,len(levels)-1)][:-4]
		done = False
	else:
		done = False
print('Playing ' + level_name)
p, width, height = utils.import_raw_level(level_name)
steps = []

original_arr = copy.deepcopy(p.arr)
original_arr[:,:,5:] = np.zeros((original_arr[:,:,5:].shape))
originalchar_loc = copy.deepcopy(p.char_loc) 


pg.init()

scr_size = scr_width, scr_height = [600, 400]

screen = pg.display.set_mode(scr_size)

clock = pg.time.Clock()

using_net = True

prediction = None
if using_net:
    netname = "deep_moveinfo"
    #netname = "deep_noinfo"
    model = utils.get_model(netname)
    query = np.zeros((1, 20, 20, 12))
    query[0,:,:,:] = copy.deepcopy(p.arr)
    utils.position_transform(query[0,:,:,:])
    prediction = model.predict(query)[0]
while not done:
    clock.tick(50)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done=True
        if event.type == pg.KEYDOWN:
            
            # Moves
            if event.key == pg.K_UP:
                if p.step_in_direction(0):
                    steps.append(0)
            if event.key == pg.K_RIGHT:
                if p.step_in_direction(1):
                    steps.append(1)
            if event.key == pg.K_DOWN:
                if p.step_in_direction(2):
                    steps.append(2)
            if event.key == pg.K_LEFT:
                if p.step_in_direction(3):
                    steps.append(3)
                
            if event.key == pg.K_r:  # R to restart
                steps = []
                p, width, height = utils.import_raw_level(level_name)
            if event.key == pg.K_u:  # U to undo
                steps = steps[:-1]
                p, width, height = utils.import_raw_level(level_name)
                for i in steps:
                    p.step_in_direction(i)
            if event.key == pg.K_s:  # S to save data
                comprehensive_arr = [original_arr, originalchar_loc, steps]
                f2 = open(outpath+level_name, "wb")
                pickle.dump(comprehensive_arr, f2)
                f2.close()
            query = np.zeros((1, 20, 20, 12))
            query[0,:,:,:] = copy.deepcopy(p.arr)
            utils.position_transform(query[0,:,:,:])
            prediction = model.predict(query)[0]
    draw(width, height, p, steps, prediction, using_net)
    
pg.quit()

