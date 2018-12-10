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

import random,util,time


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



    def getAction(self, state, numMove=3, curPoints=0):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        if numMove == 0:
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
        maxAction = ["Stop"]
        actions = {'West': -1, 'East': 1, 'North': -self.width - 1, 'South': self.width + 1, 'Stop': 0}
        for action in state.getLegalActions():
            moveTo = pacmanPos + actions[action]
            newPos = strState[moveTo] # Char Value
            newPoints = self.rewards[0]
            if newPos == '.' or newPos == 'o': # Food in position
                newPoints += self.rewards[2]
            elif newPos == 'G': # Ghost in position
                newPoints += self.rewards[1]

            newPoints = self.lookForward(state.generateSuccessor(0, action), numMove - 1, newPoints)

            if newPoints > maxPoints:
                maxPoints = newPoints
                maxAction = [action]
            elif newPoints == maxPoints:
                maxAction.append(action)

        print(maxAction)
        return random.choice(maxAction)