# What is PP1?

This is a CNN-based solver for the Flash game [Psychopath](http://k2xl.com/games/psychopath/?go=index2), which is similar to [Sokoban](https://en.wikipedia.org/wiki/Sokoban). In Psychopath, the player is on a grid and can take steps either up, down, left, or right. The goal is to get to a particular square ("winspace") in as few steps as possible. There are two types of obstacles in the way. Some squares ("unmovables") cannot be stepped on or used in any way. Other squares contain "blocks" (or "movables") which can be pushed, as long as the square behind the block is not an unmovable and doesn't contain a block. (Note: Psychopath 2 has the option of extra obstacles, but there are many Psychopath 1 style levels there as well.)

# Why not just A* search?

At first, this may sound susceptible to some variant of A*. Many of the attempts along these lines were lost with forum crashes. There is [an honors thesis](https://is.muni.cz/th/173095/fi_b/BP.pdf) (written in Czech) which was able to solve some early levels. The issue, however, is that many levels require significant backtracking, due to a structure called a "lock." Consider this level:

![9x8](https://github.com/davidspencer6174/pp1solver/blob/master/images/9x8PP1Annotated.png)

(Image modified from a screenshot on the Psychopath 2 website.) The yellow square is the character, the rainbow square is the winspace, the dark squares are unmovables, and the tan squares are blocks.

The labeled blocks form one of the four locks in the level. To gain access to the left side, it is necessary to push each of 1 and 3 to the left, and then 2 can be pushed either up or down. This may not seem bad, since the character is right next to 1, but looking ahead a bit further, it turns out that it is necessary to push 2 down rather than up, or else it will be in the way when the player tries to access the upper left (by going around to the left side of 1 and pushing it back to the right). The player is required to first go down to 3 and then return all the way back to 1. Typically, levels which challenge experienced human players require careful planning because they contain many locks which enforce such an ordering.

# The approach

This project is an attempt at using neural networks to assist in solving levels. The SolvedLevels folder contains 98 levels with optimal solutions. These files were generated using gameplay.py, which accepts as its input level code from the Psychopath 2 website (the level code files are in RawLevels) and allows the user to play the level and save movesets. The neural network takes in board positions and outputs probabilities that each possible action is optimal.

For humans, the action space is discretized by individual steps, but in the neural net, we take it to be discretized by pushes (or by reaching the winspace). This is because the quickest route to a push is easily computed without assistance from neural networks. Depending on the specific test set, we tend to achieve 68-80% test accuracy at predicting the location and direction of the optimal action.

After the network is trained, we perform a tree search. At the moment, the search is naive, ad hoc, and has some magic numbers. It maintains a queue of the positions, where the priority is a combination of the number of steps taken so far and the likelihoods for each of the steps given by the neural network. It repeatedly expands a batch of the highest-priority positions until it finds solutions.

# Results

Testing so far has been limited. The level pictured in the earlier section has an optimal solution of 63 steps. The search found a 69-step solution in 444 seconds. It made a four-step mistake in the upper right, opening the lock in the wrong direction, and a two-step mistake in the upper left, failing to find the optimal last few steps. It then found a 67-step solution in 2791 seconds, opening the lock in the upper-right in the correct order this time, but failing to take full advantage of the resulting benefits.

It's encouraging that it was able to come up with a solution to a level with some backtracking and a few locks, even if its solution was not optimal. It may be useful to compare with the results from the aforementioned honors thesis, which tested on the official game levels. This will require training the network without using any of the official game levels as input. However, we can say confidently without any further testing that performance is far below the performance of experienced human solvers, regardless of how this method compares to A*-like searches. Challenging levels for humans tend to be 20x20 and require hundreds of steps, such as the below 210-step level:

![Pentessence](https://github.com/davidspencer6174/pp1solver/blob/master/images/Pentessence.PNG)

# Technical Details

The input positions are 39x39 with 12 layers. Each position has the character centered at (19, 19) and is padded with movable blocks. The network failed to learn without centering the character; it's not clear whether this is due to a bug or whether it is necessary for the character to be centered.

Layers 0-5 encode the position: unmovable blocks, movable blocks, character, winspace, empty squares. Layer 5 has a 1 wherever the character can reach without using a push. Layers 6-10 represent locations at which the character can make a push as the next action in the directions up, right, down, and left respectively; the values are -n where n is the number of 
steps required to make the push. Layer 10 has a nonzero value if an exit can be reached at the location; its value is the maximum of layers 6-10 if it is covered by a movable and is -n if it is not covered by a movable and it takes n steps to reach the exit. Layer 11 is the sum over all previous Layer 5's, to give some notion of history (ideally, this will encourage exploration of new areas).

The output targets are one-hot vectors of length 39\*39\*4, to account for 4 possible directions of pushes in each of 39\*39 locations. A winspace entry not covered by a movable is encoded in the direction in which the step occurs in the training data.

The network at present is very simple: two convolutional layers and a dense layer, with dropout after each convolutional layer. The dataset is fairly small, so it's not clear whether performance can get much better.

The data consists of 2970 actions. After including rotations and reflections, we have 23760 data points.

# Set Up

1. Install [Anaconda](https://www.anaconda.com/distribution/) and [git](https://git-scm.com/downloads) 
2. Open Anaconda Prompt and execute "git clone https://github.com/davidspencer6174/pp1solver.git" in your desired folder
3. execute "conda env create -f environment.yml --force"
4. execute "conda activate pp1solver"
5. execute "python main.py" to train or "python gameplay.py" to play levels


# To-do list

* Try dealing with overfitting via regularization rather than dropout. Also experiment with dropout on the input instead of after the hidden layers.
* Figure out a way to produce a value net. Then take that along with the existing policy net and try some variant of MCTS. We have tried some ways to get a value net, but none are entirely convincing yet.
* Test more thoroughly to determine whether the tree search is capable of solving levels the A*-like paper cannot solve.
* Profile the code - I would expect that the forward passes are the bottleneck, but if not, it may be necessary to optimize the PushPosition class or the tree search algorithm itself.
* Produce more data.

# Acknowledgements

Many thanks to my collaborator Joshua Mu for contributing countless ideas and suggestions, helping with data generation, assisting in testing, and catching many of the bugs and data inconsistencies I introduced.

Also, credit to the following users, whose levels I used: FlashBack, hi19hi19, vanadium, ybbun, jamesandpie, Tim, stargroup100, KiggD, salamander, Anorhc, Razslo, c117, and blue. If you would prefer not to have your levels used, please feel free to contact me and I will remove them from the dataset and the repository. The levels pictured in this README ("9x8 PP1" and "Pentessence") are my own.
