#Dependencies
import pickle
import copy
import random
from utils import PushPosition
import numpy as np
import os
import sys
import pygame as pg

#Variables
ILLEGAL = -1
NORMAL = 1
PUSH = 2
WIN = 10000

num_layers = 12
size = 20
rawpath = "RawLevels/"
outpath = "AlteredLevels/"
steps=[]

#Altered import_raw_level function. Can change weights to change how it's altered.
def import_altered_raw_level(level, raw_path):
    """
    Imports a level given the level code in the official PP2 format.
    Pads the level with unmovables to make it 20x20.
    """
    f = open(raw_path+level+".txt", "r")
    level = f.readline().split()  
    metadata = level[0].split(",")
    width, height = int(metadata[2]), int(metadata[3])
    arr = np.zeros((20, 20, 10))
    arr[:,:,4] = 1
    for i in range(0, 20):
        for j in range(0, 20):
            if i >= height or j >= width:
                arr[i,j,0] = 1
                arr[i,j,4] = 0
            if i < height and j < width:
                randomvalue=np.random.random_integers(1,100)
				#5% to be block
                if randomvalue<6:
                    arr[i,j,0] = 1
                    arr[i,j,4] = 0
				#5% to immovable
                elif randomvalue<11:
                    arr[i,j,1] = 1
                    arr[i,j,4] = 0		
				#90% to remain empty 

    for item in level[1:]:
        randomvalue=np.random.random_integers(1,100)
        vals = item.split(":")[1].split(",")
        if item[0] == "w":
            arr[int(vals[1]),int(vals[0]),3] = 1
        if item[0] == "c":
            arr[int(vals[1]),int(vals[0]),2] = 1
		#85% for the immoveable/moveable block to remain the same
        if randomvalue>15:
            if item[0] == "b":
                arr[int(vals[1]),int(vals[0]),0] = 1
                arr[int(vals[1]),int(vals[0]),4] = 0
            if item[0] == "m":
                arr[int(vals[1]),int(vals[0]),1] = 1
                arr[int(vals[1]),int(vals[0]),4] = 0
		#10% for the immoveable/moveable block to switch types
        if randomvalue>=5 and randomvalue<15:
            if item[0] == "b":
                arr[int(vals[1]),int(vals[0]),1] = 1
                arr[int(vals[1]),int(vals[0]),4] = 0
            if item[0] == "m":
                arr[int(vals[1]),int(vals[0]),0] = 1
                arr[int(vals[1]),int(vals[0]),4] = 0
		#5% for the immoveable/moveable block to become blank space
    p = PushPosition(arr)
    return p, width, height

#Draw function from gameplay.py
def draw(width, height, p, steps):    #draw the grid
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
   

#List of all the levels in rawlevels
levels=os.listdir(path='RawLevels')

white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255, 255, 0)
grey = (128, 128, 128)
blue = (0, 0, 255)

done = False

#Asks User for Level Name, Provides random level if not available
print('Which level would you like to play? (hit enter for random level) ')
level_name=str(input())
if level_name+'.txt' not in levels:
    level_name=levels[np.random.random_integers(0,len(levels)-1)][:-4]
print('Playing alterations of ' + level_name)

#Asks user for how many different variations they want to see
print('How many alterations would you like to see? ')
numoftimes=int(input())
print('Yellow is the character, Blue is the goal')
#Opens pickle of altered levels, creates new list if not created
try:
    with open(outpath+level_name+' altered') as f:
        responses = pickle.load(f)
except:
    responses=[]

#Displays altered levels
for i in range(numoftimes):
    p, width, height = import_altered_raw_level(level_name, rawpath)

    original_arr = copy.deepcopy(p.arr)
    original_arr[:,:,5:] = np.zeros((original_arr[:,:,5:].shape))
    originalchar_loc = copy.deepcopy(p.char_loc) 


    pg.init()

    scr_size = scr_width, scr_height = [600, 400]

    screen = pg.display.set_mode(scr_size)

    clock = pg.time.Clock()
    while not done:
        clock.tick(50)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done=True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_y:  # Y for yes
                    responses.append(p.arr)
                    responses.append('y')
                    done=True
                if event.key == pg.K_n:  # N for no
                    responses.append(p.arr)
                    responses.append('n')
                    done=True
                if event.key == pg.K_m:  # M for maybe
                    responses.append(p.arr)
                    responses.append('n')
                    done=True
        draw(width, height, p, steps)  
    pg.quit()
    done=False
with open(outpath+level_name+' altered', 'wb') as f:
    pickle.dump(responses, f)  

