# learningAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Directions, Agent, Actions
from util import manhattanDistance
import random,time


class IRLAgent(Agent):
    """
        After running maxent irl there should be point values for each state on the map.
        Therefore, we can
    """

    def __init__(self, rewards):
        """
            rewards = [pacman move value, ghost spot value, food value]
        """
        self.rewards = rewards
        self.width = 20
        self.height = 7


    ####################################
    #    Override These Functions      #
    ####################################
    def lookForward(self, state, numMoves, curPoints):
        if numMoves == 0:
            return curPoints

        strState = str(state)[:str(state).find('Score')-1]
        strState = strState.replace('\\n', "")

        if strState.find('v') > -1:
            pacmanPos = strState.find('v')
        elif strState.find('^') > -1:
            pacmanPos = strState.find('^')
        elif strState.find('<') > -1 :
            pacmanPos = strState.find('<')
        else:
            pacmanPos = strState.find('>')

        maxPoints = -10000
        actions = {'West': -1, 'East': 1, 'North': -self.width - 1, 'South': self.width + 1, 'Stop': 0}

        for action in state.getLegalActions():
            moveTo = pacmanPos + actions[action]
            newPos = strState[moveTo] # Char Value
            newPoints = self.rewards[0]
            if newPos == '.' or newPos == 'o': # Food in position
                newPoints += self.rewards[2]
            elif newPos == 'G': # Ghost in position
                newPoints += self.rewards[1]

            newPoints = self.lookForward(state.generateSuccessor(0, action), numMoves-1, newPoints)

            if newPoints > maxPoints:
                maxPoints = newPoints

        return maxPoints


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

		# Calculate the manhattan distance to each food pellet
        foodDistances = []
        for foodPos in newFood.asList():
          foodDistances.append(manhattanDistance(newPos, foodPos))

		# Increment the score by the (food value/closest food distance)
        if len(foodDistances) > 0:
          score += self.rewards[2]/min(foodDistances)

		# Calculate how many ghosts are close to this next move
        ghostClosenessCount = 0
        for i in range(0, len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition()

			# A ghost is too close if their manhattan distance is <= 3
			# This might be too conservative
            tooClose = manhattanDistance(newPos, ghostPos) <= 2

            if tooClose and newScaredTimes[i] == 0:
                ghostClosenessCount += 1

		# Subtract 500 points for each close ghost
		# because we just really don't want to die
        # (and you lose 500 points for dying)
        score += self.rewards[1]*ghostClosenessCount

        return score

        # """
        # state: can call state.getLegalActions()
        # Choose an action and return it.
        # """
        # if numMove == 0:
        #     return curPoints

        # strState = str(state)[:str(state).find('Score')-1]
        # strState = strState.replace('\\n', "")


        # if strState.find('v') > -1:
        #     pacmanPos = strState.find('v')
        # elif strState.find('^') > -1:
        #     pacmanPos = strState.find('^')
        # elif strState.find('<') > -1 :
        #     pacmanPos = strState.find('<')
        # else:
        #     pacmanPos = strState.find('>')

        # maxPoints = -10000
        # maxAction = ["Stop"]
        # actions = {'West': -1, 'East': 1, 'North': -self.width - 1, 'South': self.width + 1, 'Stop': 0}
        # for action in state.getLegalActions():
        #     moveTo = pacmanPos + actions[action]
        #     newPos = strState[moveTo] # Char Value
        #     newPoints = self.rewards[0]
        #     if newPos == '.' or newPos == 'o': # Food in position
        #         newPoints += self.rewards[2]
        #     elif newPos == 'G': # Ghost in position
        #         newPoints += self.rewards[1]

        #     newPoints = self.lookForward(state.generateSuccessor(0, action), numMove - 1, newPoints)

        #     if newPoints > maxPoints:
        #         maxPoints = newPoints
        #         maxAction = [action]
        #     elif newPoints == maxPoints:
        #         maxAction.append(action)

        # print(maxAction)
        # return random.choice(maxAction)