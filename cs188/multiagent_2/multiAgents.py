# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        distanceFromGhosts = 0
        scaredBonus = 0
        distanceFromFoods = 0
        # always give pacman a food as destination
        for food in newFood.asList():
            distanceFromFoods += manhattanDistance(newPos, food)
            break
        # distance from ghosts, when not scared, the further the better, but distances further than a certain number are considered the same(safe for pacman)
        # when too close to ghost, give it a penalty score to keep pacman from being eaten
        # when ghosts are scared distance from ghosts are not considered
        safeDistance = 6
        tooCloseDistance = 2
        for i in range(len(newGhostStates)):
            distance = manhattanDistance(newPos, newGhostStates[i].getPosition())
            if newScaredTimes[i] <= 0:
                if distance > tooCloseDistance and distance < safeDistance:
                    distanceFromGhosts += distance
                elif distance >= safeDistance:
                    distanceFromGhosts += safeDistance
                else:
                    distanceFromGhosts -= 10
            else:
                scaredBonus += 50
        return successorGameState.getScore() + distanceFromGhosts - distanceFromFoods + scaredBonus

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
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        stepsLimit = gameState.getNumAgents() * self.depth
        def minimax(gameState, agentIndex, step):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif step == stepsLimit:
                return min([self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex)])
            else:
                if agentIndex == 0:
                    return max([minimax(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numAgents, step+1) for action in gameState.getLegalActions(agentIndex)])
                else:
                    return min([minimax(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numAgents, step+1) for action in gameState.getLegalActions(agentIndex)])

	maxEvaluate = -999999
	for action in gameState.getLegalActions(0):
	    evaluate = minimax(gameState.generateSuccessor(0, action), 1, 2)
	    if evaluate > maxEvaluate:
	        maxEvaluate, result = evaluate, action
        return result
                
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        stepsLimit = gameState.getNumAgents() * self.depth
        def selectValueFunction(agentIndex):
            if agentIndex == 0: return maxValue
            else: return minValue
        
        def maxValue(gameState, alpha, beta, agentIndex, step):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = -999999
            nextAgentIndex = (agentIndex+1)%numAgents
            nextValueFunction = selectValueFunction(nextAgentIndex)
            for action in gameState.getLegalActions(agentIndex):
                if step == stepsLimit:
                    v = max(v, self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)))
                else:
                    v = max(v, nextValueFunction(gameState.generateSuccessor(agentIndex, action), alpha, beta, nextAgentIndex, step+1))
                if v > beta: 
                    return v
                alpha = max(v, alpha)
            return v

        def minValue(gameState, alpha, beta, agentIndex, step):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = 999999
            nextAgentIndex = (agentIndex+1)%numAgents
            nextValueFunction = selectValueFunction(nextAgentIndex)
            for action in gameState.getLegalActions(agentIndex):
                if step == stepsLimit:
                    v = min(v, self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)))
                else:
                    v = min(v, nextValueFunction(gameState.generateSuccessor(agentIndex, action), alpha, beta, nextAgentIndex, step+1))
                if v < alpha:
                    return v
                beta = min(v, beta)
            return v

        alpha, beta = -999999, 999999
        for action in gameState.getLegalActions(0):
            evaluate = minValue(gameState.generateSuccessor(0, action), alpha, beta, 1, 2)
            if evaluate > alpha:
                alpha, result = evaluate, action
        return result

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
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        stepsLimit = gameState.getNumAgents() * self.depth
        def expectimax(gameState, agentIndex, step):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif step == stepsLimit:
                return float(sum([self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex)]))/float(len(gameState.getLegalActions(agentIndex)))
            else:
                if agentIndex == 0:
                    return max([expectimax(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numAgents, step+1) for action in gameState.getLegalActions(agentIndex)])
                else:
                    return float(sum([expectimax(gameState.generateSuccessor(agentIndex, action), (agentIndex+1)%numAgents, step+1) for action in gameState.getLegalActions(agentIndex)]))/float(len(gameState.getLegalActions(agentIndex)))
            
        maxEvaluate = -999999
        for action in gameState.getLegalActions(0):
            evaluate = expectimax(gameState.generateSuccessor(0, action), 1, 2)
            if evaluate > maxEvaluate:
                maxEvaluate, result = evaluate, action
        return result

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      0.we always give pacman a food as destination
      1.distance from ghosts, when not scared, the further the better,
        but distances further than a certain number are considered the same(safe for pacman)
        thus pacman won`t always trying to get away from ghosts
      2.when too close to ghost, give it a penalty score to keep pacman from being eaten
      3.when ghosts are scared distance from ghosts are not considered
      4.we give a bonus when ghosts are scared to encourage Pacman to eat pellets
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    distanceFromGhosts = 0
    scaredBonus = 0
    distanceFromFoods = 0
    for food in foods.asList():
        distanceFromFoods += manhattanDistance(pos, food)
        break
    safeDistance = 6
    tooCloseDistance = 2
    for i in range(len(ghostStates)):
        distance = manhattanDistance(pos, ghostStates[i].getPosition())
        if scaredTimes[i] <= 0:
            if distance > tooCloseDistance and distance < safeDistance:
                distanceFromGhosts += distance
            elif distance >= safeDistance:
                distanceFromGhosts += safeDistance
            else:
                distanceFromGhosts -= 10
        else:
            scaredBonus += 50
    return currentGameState.getScore() + distanceFromGhosts - distanceFromFoods + scaredBonus

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

