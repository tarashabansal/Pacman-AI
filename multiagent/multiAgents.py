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
		currentFood = currentGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		score = successorGameState.getScore()
		new_ghost_positions = successorGameState.getGhostPositions()
		current_food_list = currentFood.asList()
		new_food_list = newFood.asList()
		closest_food = float('+Inf')
		closest_ghost = float('+Inf')
		add_score = 0

		"""
		Rewarding the agent for moving (choosing an action towards food) towards food pellets.
		Calculating the manhattan distance to all the food pellets for the successor game state.

		If pacman agent is on a food pellet we add 10.0 to our current_score, afterwards we
		calculate the distance to the closest food and award our agent a value 10.0 divided
		by the distance to the closest food. The closer our agent is to the closest pellet
		the larger the score. Then we subtract the total available food for the state multiplied
		by 4.0 enforcing our agent to minimize the remaining food value and we add the score from
		the pellets we already cleared. After that we calculate the distance from the closest ghost,
		if the ghost is closer than 2 spots on the field we decrease the score by a big number in
		order to enforce our agent to move away from ghosts.
		"""

		if newPos in current_food_list:
			add_score += 10.0

		distance_from_food = [manhattanDistance(newPos, food_position) for food_position in new_food_list]
		total_available_food = len(new_food_list)
		if len(distance_from_food):
			closest_food = min(distance_from_food)

		score += 10.0 / closest_food  - 4.0 * total_available_food + add_score

		# ? TODO: Write some comments about the implementation.

		for ghost_position in new_ghost_positions:
			distance_from_ghost = manhattanDistance(newPos, ghost_position)
			closest_ghost = min([closest_ghost, distance_from_ghost])

		if closest_ghost < 2:
			score -= 50.0

		return score

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
		# return self.minimax(gameState=gameState)
		best_action = self.max_value(gameState=gameState, depth=0, agent_idx=0)[1]
		return best_action
		# util.raiseNotDefined()

	def is_terminal_state(self, gameState, depth, agent_idx):
		"""
		Helper function to determine if we reached a leaf node in the state search tree

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]

		Returns:
			[boolean] -- [Win state from set of win states given in the initialization of the problem.]
			[boolean] -- [Loss state from set of loss states given in the initialization of the problem.]
			[list] -- [List of legal actions corresponding to the agent index given from the defaultdict of stateToActions.]
			[int] -- [Depth to expand our search tree.]
		"""

		if gameState.isWin():
			return gameState.isWin()
		elif gameState.isLose():
			return gameState.isLose()
		elif gameState.getLegalActions(agent_idx) is 0:
			return gameState.getLegalActions(agent_idx)
		elif depth >= self.depth * gameState.getNumAgents():
			return self.depth

	def max_value(self, gameState, depth, agent_idx):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the maximizing
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. In order to get the value and the corresponding action we need to create an iterable object such as a list
		and specify the key with which we make the comparison for the maximum value which is the float value in the first
		position of the tuple hence the idx[0].

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]

		Returns:
			[tuple(float, string)] -- [The maximum minimax value for a gameState node and the corresponding action]
		"""

		value = (float('-Inf'), None)
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = (depth + 1) % number_of_agents
			value = max([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player), action)], key=lambda idx: idx[0])
		return value

	def min_value(self, gameState, depth, agent_idx):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the minimizing
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. In order to get the value and the corresponding action we need to create an iterable object such as a list
		and specify the key with which we make the comparison for the minimum value which is the float value in the first
		position of the tuple hence the idx[0].

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]

		Returns:
			[tuple(float, string)] -- [The minimum minimax value for a gameState node and the corresponding action]
		"""

		value = (float('+Inf'), None)
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = (depth + 1) % number_of_agents
			value = min([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player), action)], key=lambda idx: idx[0])
		return value

	def value(self, gameState, depth, agent_idx):
		"""
		Helper function that acts as a dispatcher to the above functions. It determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
		and traverses the tree to the leaves and backs up the state's utility value.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]

		Returns:
			[float] -- [State's utility value.]
		"""

		if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
			return self.evaluationFunction(gameState)
		elif agent_idx is 0:
			return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]
		else:
			return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		alpha = float('-Inf')
		beta = float('+Inf')
		depth = 0
		best_action = self.max_value(gameState=gameState, depth=depth, agent_idx=0, alpha=alpha, beta=beta)
		return best_action[1]
		# util.raiseNotDefined()

	def is_terminal_state(self, gameState, depth, agent_idx):
		"""
		Helper function to determine if we reached a leaf node in the state search tree

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]

		Returns:
			[boolean] -- [Win state from set of win states given in the initialization of the problem.]
			[boolean] -- [Loss state from set of loss states given in the initialization of the problem.]
			[list] -- [List of legal actions corresponding to the agent index given from the defaultdict of stateToActions.]
			[int] -- [Depth to expand our search tree.]
		"""

		if gameState.isWin():
			return gameState.isWin()
		elif gameState.isLose():
			return gameState.isLose()
		elif gameState.getLegalActions(agent_idx) is 0:
			return gameState.getLegalActions(agent_idx)
		elif depth >= self.depth * gameState.getNumAgents():
			return self.depth

	def max_value(self, gameState, depth, agent_idx, alpha, beta):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the maximizing
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. In order to get the value and the corresponding action we need to create an iterable object such as a list
		and specify the key with which we make the comparison for the maximum value which is the float value in the first
		position of the tuple hence the idx[0]. Additionally by using the alpha factor we can prune whole game-state subtrees.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]
			alpha {[float]} -- [Float value which represents the minimum score that the maximizing player is assured of.]
			beta {[float]} -- [Float value which represents the maximum score that the minimizing player is assured of.]

		Returns:
			[tuple(float, string)] -- [The maximum minimax value for a gameState node and the corresponding action.]
		"""


		value = (float('-Inf'), None)
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = expand % number_of_agents
			value = max([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
			if value[0] > beta:
				return value
			alpha = max(alpha, value[0])
		return value

	def min_value(self, gameState, depth, agent_idx, alpha, beta):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the minimizing
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. In order to get the value and the corresponding action we need to create an iterable object such as a list
		and specify the key with which we make the comparison for the minimum value which is the float value in the first
		position of the tuple hence the idx[0]. Additionally by using the beta factor we can prune whole game-state subtrees.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]
			alpha {[float]} -- [Float value which represents the minimum score that the maximizing player is assured of.]
			beta {[float]} -- [Float value which represents the maximum score that the minimizing player is assured of.]

		Returns:
			[tuple(float, string)] -- [The maximum minimax value for a gameState node and the corresponding action.]
		"""


		value = (float('+Inf'), None)
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = expand % number_of_agents
			value = min([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
			if value[0] < alpha:
				return value
			beta = min(beta, value[0])
		return value

	def value(self, gameState, depth, agent_idx, alpha, beta):
		"""
		Helper function that acts as a dispatcher to the above functions. It determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
		and traverses the tree to the leaves and backs up the state's utility value.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [description]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]
			alpha {[float]} -- [Float value which represents the minimum score that the maximizing player is assured of.]
			beta {[float]} -- [Float value which represents the maximum score that the minimizing player is assured of.]

		Returns:
			[float] -- [State's utility value.]
		"""


		if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
			return self.evaluationFunction(gameState)
		elif agent_idx is 0:
			return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx,alpha=alpha, beta=beta)[0]
		else:
			return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx, alpha=alpha, beta=beta)[0]

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
		best_action = self.max_value(gameState=gameState, depth=0, agent_idx=0)[1]
		return best_action
		# util.raiseNotDefined()

	def is_terminal_state(self, gameState, depth, agent_idx):
		"""
		Helper function to determine if we reached a leaf node in the state search tree

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree. This is the maximum depth to expand to.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions.]

		Returns:
			[boolean] -- [Win state from set of win states given in the initialization of the problem.]
			[boolean] -- [Loss state from set of loss states given in the initialization of the problem.]
			[list] -- [List of legal actions corresponding to the agent index given from the defaultdict of stateToActions.]
			[int] -- [Depth to expand our search tree.]
		"""

		if gameState.isWin():
			return gameState.isWin()
		elif gameState.isLose():
			return gameState.isLose()
		elif gameState.getLegalActions(agent_idx) is 0:
			return gameState.getLegalActions(agent_idx)
		elif depth >= self.depth * gameState.getNumAgents():
			return self.depth

	def max_value(self, gameState, depth, agent_idx):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the maximizing
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. In order to get the value and the corresponding action we need to create an iterable object such as a list
		and specify the key with which we make the comparison for the maximum value which is the float value in the first
		position of the tuple hence the idx[0].

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]
		Returns:
			[tuple(float, string)] -- [The maximum minimax value for a gameState node and the corresponding action.]
		"""

		value = (float('-Inf'), None)
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = (depth + 1) % number_of_agents
			value = max([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player), action)], key=lambda idx: idx[0])
		return value

	def expected_value(self, gameState, depth, agent_idx):
		"""
		Helper function to go through the whole game-state tree, all the way to the leaves, to determine the average
		backed up value of a state. The implementation is based on the slides of the Berkeley CS 188: Artificial Intelligence
		class. Here we make use and calculate the simple average of the children and not the weighted average because we consider
		all the events to be equally probable.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]
		Returns:
			[float] -- [The average expected utilities value of children]
		"""
		value = list()
		legal_actions = gameState.getLegalActions(agent_idx)
		for action in legal_actions:
			successor_state = gameState.generateSuccessor(agent_idx, action)
			number_of_agents = gameState.getNumAgents()
			expand = depth + 1
			current_player = (depth + 1) % number_of_agents
			value.append(self.value(gameState=successor_state, depth=expand, agent_idx=current_player))
		expected_value = sum(value) / len(value)
		return expected_value


	def value(self, gameState, depth, agent_idx):
		"""
		Helper function that acts as a dispatcher to the above functions. It determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
		and traverses the tree to the leaves and backs up the state's utility value.

		Arguments:
			gameState {[MultiagentTreeState object]} -- [Object that represents the state of the problem.]
			depth {[int]} -- [The depth of the search tree in which we expand.]
			agent_idx {[int]} -- [Agent index (pacman, ghosts) to test for remaining legal actions, and calculate node values.]
		"""

		if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
			return self.evaluationFunction(gameState)
		elif agent_idx is 0:
			return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]
		else:
			return self.expected_value(gameState=gameState, depth=depth, agent_idx=agent_idx)


def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: This evaluation function is a linear combination of the below
	  			   features:
					 --- distance to closest food pellet
					 --- distance to closest enemy ghost
					 --- distance to closest scared ghost
					 --- remaining food capsules
					 --- remaining food pellets
					 --- current score

					First we calculate the current game-state score we subtract
					for each feature the minimum distance or the remaining multitude
					multiplied by a weight according to the importance of the feature.

					For the closest food pellet we subtract the minimum distance. The
					larger the distance from the closest food pellet the bigger the
					penalty. Therefore the pacman agent will prefer actions that
					minimize this value by moving closer to food pellets.

					For the closest enemy ghost we subtract the inverse minimum
					distance multiplied by 2.0. This means that the farther the
					pacman agent is to a non-scared ghost the less negative the
					score is. Therefore the pacman agent will prefer actions that
					minimize this value by moving away from non-scared ghosts.

					For the closest scared ghost we subtract the minimum distance
					multiplied by 3.0. This means that the closer our agent is to
					a scared ghost the less negative the score will be, enforcing
					the agent to choose action towards scared ghosts. Moreover by
					eating a scared ghost we get more points.

					For the food capsules i multiply the number of remaining capsules
					by a big number so pacman should try anf minimize the number of
					capsules eating them as he passes by them.

					Accordingly for the remaining food i multiply and subtract by the
					number of the remaing food pellets in order for the pacman agent
					to try and minimize the number of remaining food pellets so we
					can win the stage.
	"""
	pacman_position = currentGameState.getPacmanPosition()
	food_positions = currentGameState.getFood().asList()
	capsules_positions = currentGameState.getCapsules()
	ghost_positions = currentGameState.getGhostPositions()
	ghost_states = currentGameState.getGhostStates()
	scared_ghosts_timer = [ghost_state.scaredTimer for ghost_state in ghost_states]
	remaining_food = len(food_positions)
	remaining_capsules = len(capsules_positions)
	scared_ghosts = list()
	enemy_ghosts = list()
	enemy_ghost_positions = list()
	scared_ghosts_positions = list()
	score = currentGameState.getScore()

	closest_food = float('+Inf')
	closest_enemy_ghost = float('+Inf')
	closest_scared_ghost = float('+Inf')

	distance_from_food = [manhattanDistance(pacman_position, food_position) for food_position in food_positions]
	if len(distance_from_food) is not 0:
		closest_food = min(distance_from_food)
		score -= 1.0 * closest_food

	for ghost in ghost_states:
		if ghost.scaredTimer is not 0:
			enemy_ghosts.append(ghost)
		else:
			scared_ghosts.append(ghost)

	for enemy_ghost in enemy_ghosts:
		enemy_ghost_positions.append(enemy_ghost.getPosition())

	if len(enemy_ghost_positions) is not 0:
		distance_from_enemy_ghost = [manhattanDistance(pacman_position, enemy_ghost_position) for enemy_ghost_position in enemy_ghost_positions]
		closest_enemy_ghost = min(distance_from_enemy_ghost)
		score -= 2.0 * (1 / closest_enemy_ghost)

	for scared_ghost in scared_ghosts:
		scared_ghosts_positions.append(scared_ghost.getPosition())

	if len(scared_ghosts_positions) is not 0:
		distance_from_scared_ghost = [manhattanDistance(pacman_position, scared_ghost_position) for scared_ghost_position in scared_ghosts_positions]
		closest_scared_ghost = min(distance_from_scared_ghost)
		score -= 3.0 * closest_scared_ghost

	score -= 20.0 * remaining_capsules
	score -= 4.0 * remaining_food
	return score
	# util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction