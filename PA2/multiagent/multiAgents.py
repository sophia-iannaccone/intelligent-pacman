# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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

        score = 0

        def find_food(agent_pos, f_pos):
            food_dist = []
            for f in f_pos:
                food_dist.append(util.manhattanDistance(f, agent_pos))
            return min(food_dist)

        x, y = newPos[0], newPos[1]
        if currentGameState.getFood()[x][y]:
            score = score + 1

        food_pos = []
        for food in newFood.asList():
            if newFood:
                food_pos.append(food)

        next_food = find_food(newPos, food_pos)
        curr_food = find_food(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
        if next_food < curr_food:
            score = (1 / (next_food - curr_food)) * 3
        else:
            score = score - 15

        for ghost in newGhostStates:
            ghost_dist = util.manhattanDistance(ghost.getPosition(), newPos)
            if ghost_dist > 1:
                score = score - 20

        return score + currentGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def gameOver(gameState):
            return gameState.isWin() or gameState.isLose()

        def minimax(gameState, agent, depth):
            if depth == self.depth or gameOver(gameState):
                return self.evaluationFunction(gameState)

            # Pacman AKA Max turn
            if agent == 0:
                v = float('-inf')
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    v = max(v, minimax(successor, 1, depth))
                return v

            # Ghosts AKA Min turn
            else:
                next_agent = agent + 1
                # No more ghosts, end ghost/min turn
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                # Next turn is Pacman, increase depth
                if next_agent == 0:
                    depth = depth + 1

                v = float('inf')
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    v = min(v, minimax(successor, next_agent, depth))
                return v

        max_v = float('-inf')
        opt_move = Directions.STOP
        moves = gameState.getLegalActions(0)

        for m in moves:
            next_state = gameState.generateSuccessor(0, m)
            curr_max = minimax(next_state, 1, 0)
            if curr_max > max_v:
                max_v = curr_max
                opt_move = m
        return opt_move

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def gameOver(gameState):
            return gameState.isWin() or gameState.isLose()

        def alphabeta(gameState, agent, depth, alpha, beta):
            if depth == self.depth or gameOver(gameState):
                return self.evaluationFunction(gameState)

            if agent == 0:
                v = float('-inf')
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    v = max(v, alphabeta(successor, 1, depth, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                if next_agent == 0:
                    depth = depth + 1

                v = float('inf')
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    v = min(v, alphabeta(successor, next_agent, depth, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        alpha = float('-inf')
        beta = float('inf')
        opt_move = Directions.STOP
        moves = gameState.getLegalActions(0)

        for m in moves:
            next_state = gameState.generateSuccessor(0, m)
            curr_max = alphabeta(next_state, 1, 0, alpha, beta)
            if curr_max > alpha:
                alpha = curr_max
                opt_move = m
        return opt_move

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def gameOver(gameState):
            return gameState.isWin() or gameState.isLose()

        def probability(gameState, agent):
            num_moves = len(gameState.getLegalActions(agent))
            return 1 / num_moves

        def expectimax(gameState, agent, depth):
            if depth == self.depth or gameOver(gameState):
                return self.evaluationFunction(gameState)

            if agent == 0:
                v = float('-inf')
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    v = max(v, expectimax(successor, 1, depth))
                return v
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                if next_agent == 0:
                    depth = depth + 1

                v = 0
                for act in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, act)
                    num_moves = len(gameState.getLegalActions(agent))
                    p = 1/ num_moves
                    v = v + (p * expectimax(successor, next_agent, depth))
                return v

        opt_val = float('-inf')
        opt_move = Directions.STOP
        moves = gameState.getLegalActions(0)

        for m in moves:
            next_state = gameState.generateSuccessor(0, m)
            curr_max = expectimax(next_state, 1, 0)
            if curr_max > opt_val:
                opt_val = curr_max
                opt_move = m
        return opt_move

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
