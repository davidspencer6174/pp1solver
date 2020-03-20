# What is PP1?

This is a CNN-based solver for the Flash game [Psychopath](http://k2xl.com/games/psychopath/?go=index2), which is similar to [Sokoban](https://en.wikipedia.org/wiki/Sokoban). In Psychopath, the player is on a grid and can take steps either up, down, left, or right. The goal is to get to a particular square ("winspace") in as few steps as possible. There are two types of obstacles in the way. Some squares ("unmovables") cannot be stepped on or used in any way. Other squares contain "blocks" (or "movables") which can be pushed, as long as the square behind the block is not an unmovable and doesn't contain a block. (Note: Psychopath 2 has the option of extra obstacles, but there are many Psychopath 1 style levels there as well.)

# Why not just A* search?

At first, this may sound susceptible to some variant of A*. Many of the attempts along these lines were lost with forum crashes. There is [an honors thesis](https://is.muni.cz/th/173095/fi_b/BP.pdf) (written in Czech) which was able to solve some early levels. The issue, however, is that many levels require significant backtracking, due to a structure called a "lock." Therefore, coming up with a useful admissible heuristic is difficult. Although Manhattan distance is admissible, it rarely helps with pruning because stepcounts are often very large relative to the levels' dimensions. Consider this level:

![9x8](https://github.com/davidspencer6174/pp1solver/blob/master/images/9x8PP1Annotated.png)

(Image modified from a screenshot on the Psychopath 2 website.) The yellow square is the character, the rainbow square is the winspace, the dark squares are unmovables, and the tan squares are blocks.

The labeled blocks form one of the four locks in the level. To gain access to the left side, it is necessary to push each of 1 and 3 to the left, and then 2 can be pushed either up or down. This may not seem bad, since the character is right next to 1, but looking ahead a bit further, it turns out that it is necessary to push 2 down rather than up, or else it will be in the way when the player tries to access the upper left (by going around to the left side of 1 and pushing it back to the right). The player is required to first go down to 3 and then return all the way back to 1. Typically, levels which challenge experienced human players require careful planning because they contain many locks which enforce such an ordering. This can lead to very long paths.

# The approach

This project is an attempt at using neural networks to assist in solving levels. The SolvedLevels folder contains 98 levels with optimal solutions. These files were generated using gameplay.py, which accepts as its input level code from the Psychopath 2 website (the level code files are in RawLevels) and allows the user to play the level and save movesets. The policy neural network takes in board positions and outputs probabilities that each possible action is optimal.

For humans, the action space is discretized by individual steps, but in the policy net, we take it to be discretized by pushes (or by reaching the winspace). This is because the quickest route to a push is easily computed without assistance from neural networks. Depending on the specific test set, we tend to achieve 68-80% test accuracy at predicting the location and direction of the optimal action.

After the network is trained, we perform a tree search. At the moment, the search is naive, ad hoc, and has some magic numbers. It maintains a queue of the positions, where the priority is a combination of the number of steps taken so far and the likelihoods for each of the steps given by the neural network. It repeatedly expands a batch of the highest-priority positions until it finds solutions.

A high-priority next step is coming up with some way to detect which positions are more promising. This would give both a policy net and a value net, which could allow for much more powerful tree searches.

# Results

We used some of the official levels from the original game as the test set and many user-generated levels as the training set. Due to limited computational resources, we haven't been able to devote a lot of time to thorough tests to properly determine which of the networks we've generated is best. However, we can say that various networks have found optimal solutions to the large majority of the first 40 levels, including some levels which were not solved by the honors which used A* with heuristics. We can't say for sure that the neural network approach outperforms A*, though, since the honors thesis is quite old and used a much weaker computer than we have access to.

One network was able to find an optimal solution to the level pictured in the earlier section (which was also held out of the training data). However, it did not perform as well as most other networks on the official levels. It's encouraging that a network was able to come up with an optimal solution to a level with some backtracking and a few locks.

Regardless of whether the neural network approach we have so far outperforms A*, we can say confidently without any further testing that performance is far below the performance of experienced human solvers. Challenging levels for humans tend to be 20x20 and require hundreds of steps, such as the below 210-step level:

![Pentessence](https://github.com/davidspencer6174/pp1solver/blob/master/images/Pentessence.PNG)

# Technical Details

The input positions are 20x20 with 12 layers. Layers 0-5 encode the position: unmovable blocks (0 where present, 1 otherwise), movable blocks, character, winspace, empty squares (each 1 where present, 0 otherwise). (The unmovable blocks are 0 where present because we pad convolutions with 0's, and the border should be viewed as all unmovables.) Layer 5 has a 1 wherever the character can reach without using a push. Layers 6-10 represent locations at which the character can make a push as the next action in the directions up, right, down, and left respectively; the values are -n where n is the number of steps required to make the push. Layer 10 has a nonzero value if an exit can be reached at the location; its value is the maximum of layers 6-10 if it is covered by a movable and is -n if it is not covered by a movable and it takes n steps to reach the exit. Layer 11 is the sum over all previous Layer 5's, to give some notion of history (ideally, this will encourage exploration of new areas). The position_transform function in utils.py can be used to disable or transform input layers.

The output targets are one-hot vectors of length 20\*20\*4, to account for 4 possible directions of pushes in each of 20\*20 locations. A winspace entry not covered by a movable is encoded in the direction in which the step occurs in the training data.

Currently, our best network structure is a twelve-layer convolutional network with a dense output layer. It uses batch normalization, dropout, and l2 regularization.

The dataset is fairly small, so we augment it by including several transformations of the data (which presumably hurts the quality of the data). The data consists of 2970 actions. After including rotations, reflections, and translations (where applicable), we have enough data points that we can no longer hold the whole data in memory.

# Set Up

1. Install [Anaconda](https://www.anaconda.com/distribution/) and [git](https://git-scm.com/downloads) 
2. Open Anaconda Prompt and execute "git clone https://github.com/davidspencer6174/pp1solver.git" in your desired folder
3. execute "conda env create -f environment.yml --force"
4. execute "conda activate pp1solver"
5. execute "python main.py" to train or "python gameplay.py" to play levels


# To do: solvability network

This phase of the project is in progress. We would like to have a network that can estimate how likely a position is to still be solvable. To generate data for such a network, we take a prefix of a level's true solution, make a few moves from that position (probabilistically according to the policy net), and then have the program attempt to determine whether the position has a solution. It does so via Monte Carlo rollouts. Hopefully, Monte Carlo rollouts using the policy network are adequate to determine with reasonable accuracy whether a position is solvable.

# Acknowledgements

Many thanks to my collaborator Joshua Mu for contributing countless ideas and suggestions, helping with data generation, assisting in testing, and catching many of the bugs and data inconsistencies I introduced.

Also, credit to the following users, whose levels I used: FlashBack, hi19hi19, vanadium, ybbun, jamesandpie, Tim, stargroup100, KiggD, salamander, Anorhc, Razslo, c117, and blue. If you would prefer not to have your levels used, please feel free to contact me and I will remove them from the dataset and the repository. The levels pictured in this README ("9x8 PP1" and "Pentessence") are my own.
