#Dependencies
import pickle
import copy
import random
from utils import PushPosition, import_raw_level
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

expected_changes = np.array([[0, 5, 8],
                             [5, 0, 8],
                             [5, 5, 0]])

def import_altered_raw_level(level, raw_path, change):
    """
    Randomly perturbs a level so that the user can decide whether
    it's legal.  Will be used as input for the solvability net.
    The expected changes matrix gives the expected number
    of times that (unmovable, movable, empty) should be changed
    into (unmovable, movable, empty).
    The character spot is never changed.  If the winspace spot
    attempts to change to an unmovable, the change will be discarded.
    """
    p, width, height = import_raw_level(level, raw_path)
    arr = copy.deepcopy(p.arr)
    unmovables = max(np.sum(arr[0:height,0:width,0]), 1)
    movables = max(np.sum(arr[0:height,0:width,1]), 1)
    empties = max(np.sum(arr[0:height,0:width,4]), 1)
    print(unmovables)
    print(movables)
    print(empties)
    tr_mat = np.array([[0, change[0, 1]/unmovables, change[0, 2]/unmovables],
                       [change[1, 0]/movables, 0, change[1, 2]/movables],
                       [change[2, 0]/empties, change[2, 1]/empties, 0]])
    for row in range(3):
        tr_mat[row, row] = 1 - np.sum(tr_mat[row,:])
    for i in range(height):
        for j in range(width):
            r = random.random()
            if p.is_unmovable(i, j):
                if r > tr_mat[0, 0] and r <= tr_mat[0, 0] + tr_mat[0, 1]:
                    arr[i, j, 0] = 0
                    arr[i, j, 1] = 1
                if r > tr_mat[0, 0] + tr_mat[1, 1]:
                    arr[i, j, 0] = 0
                    arr[i, j, 4] = 1
            if p.is_movable(i, j):
                if r <= tr_mat[1, 0] and not p.is_win(i, j):
                    arr[i, j, 1] = 0
                    arr[i, j, 0] = 1
                if r > tr_mat[1, 0] + tr_mat[1, 1]:
                    arr[i, j, 1] = 0
                    arr[i, j, 4] = 1
            if p.is_empty(i, j):
                if r <= tr_mat[2, 0] and not p.is_win(i, j):
                    arr[i, j, 4] = 0
                    arr[i, j, 0] = 1
                if r <= tr_mat[2, 0] + tr_mat[2, 1]:
                    arr[i, j, 4] = 0
                    arr[i, j, 1] = 1
    return PushPosition(arr), width, height

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
    
nos = 0
yeses = 0

i = 0
#Displays altered levels
while i < numoftimes:
    p, width, height = import_altered_raw_level(level_name, rawpath,
                                                expected_changes)

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
                    if yeses > nos + 3:
                        done = True
                        i -= 1
                        break
                    yeses += 1
                    responses.append(p.arr)
                    responses.append('y')
                    done=True
                if event.key == pg.K_n:  # N for no
                    if nos > yeses + 3:
                        done = True
                        i -= 1
                        break
                    nos += 1
                    responses.append(p.arr)
                    responses.append('n')
                    done=True
                if event.key == pg.K_m:  # M for maybe
                    responses.append(p.arr)
                    responses.append('n')
                    done=True
        draw(width, height, p, steps)  
    pg.quit()
    i += 1
    done=False
with open(outpath+level_name+' altered', 'wb') as f:
    pickle.dump(responses, f)  

