# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        totalSum = self.total()

        if totalSum != 0:
            for key in self.keys():
                # print "@@@@@self[key] ", self[key]
                val = self[key]
                val = val / totalSum
                self[key] = val
                # print "@@@@self[key] ", self[key]
            # print "@@@@self ", self
        return self

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        import random
    
        # Get the total sum of weights
        total_weight = sum(self.values())
        
        # Generate a random number between 0 and the total weight
        random_value = random.uniform(0, total_weight)
        
        # Iterate over keys and associated weights to find the sampled key
        cumulative_weight = 0
        for key, weight in self.items():
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return key
        
        # This code should not be reached under normal circumstances
        raise ValueError("Sampling error: Unable to sample a key from the distribution")



class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        if noisyDistance == None and ghostPosition == jailPosition:
            return 1
        elif noisyDistance == None and ghostPosition != jailPosition:
            return 0
        elif noisyDistance != None and ghostPosition == jailPosition:
            return 0
        else:
            return busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"

        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        self.beliefs.normalize()

        #print "@@@", self.beliefs

        for ghostPos in self.allPositions:
            self.beliefs[ghostPos] = self.beliefs[ghostPos] * self.getObservationProb(observation, gameState.getPacmanPosition(), ghostPos, self.getJailPosition())

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"

        beliefs_copy = self.beliefs.copy()

        # find all the finalPosDistros at the beginning
        finalPosDistros = {}
        for initGhostPos in self.allPositions:
            finalPosDistr = self.getPositionDistribution(gameState, initGhostPos)
            finalPosDistros[initGhostPos] = finalPosDistr


        for finalGhostPos in self.allPositions:
            prob = 0
            for initGhostPos in self.allPositions:
                # print "@@@ init ghost pos", initGhostPos
                # print "@@@ final ghost pos", finalGhostPos
                # print "@@@ beliefs", self.beliefs[finalGhostPos]
                # print "@@@ posdistr", self.getPositionDistribution(gameState, initGhostPos)[finalGhostPos]
                prob = prob + self.beliefs[initGhostPos] * finalPosDistros[initGhostPos][finalGhostPos]

            beliefs_copy[finalGhostPos] = prob

        self.beliefs = beliefs_copy

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        flag = 0
        while flag == 0:
            for item in self.legalPositions:
                if len(self.particles) == self.numParticles:
                    flag = 1
                    break
                self.particles.append(item)


    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
       
        pacmanPos = gameState.getPacmanPosition()
        jailPos = self.getJailPosition()
        
        # Create a list of weights for each particle
        weights = []
        for particle in self.particles:
            ghostPos = particle  # Each particle represents a ghost position
            observationProb = self.getObservationProb(observation, pacmanPos, ghostPos, jailPos)
            weights.append(observationProb)
        
        # Normalize weights to form a probability distribution
        total_weight = sum(weights)
        if total_weight == 0:
            # Special case: All weights are zero, reinitialize particles uniformly
            self.initializeUniformly(gameState)
        else:
            normalized_weights = [w / total_weight for w in weights]

            # Resample particles based on the normalized weights
            new_particles = []
            num_particles = len(self.particles)
            cumulative_prob = 0.0
            for _ in range(num_particles):
                # Choose a random value between 0 and 1
                random_value = random.random()
                cumulative_prob = 0.0
                for index in range(num_particles):
                    cumulative_prob += normalized_weights[index]
                    if random_value < cumulative_prob:
                        # Select this particle based on the weight
                        new_particles.append(self.particles[index])
                        break
            
            # Update particles to the newly resampled particles
            self.particles = new_particles


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        newParticles = []
        
        for oldPos in self.particles:
            # Get the distribution of new positions given the old position
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            
            # Sample a new position from the distribution
            newPos = newPosDist.sample()
            
            # Append the new position to the list of updated particles
            newParticles.append(newPos)
        
        # Update self.particles with the new list of particles
        self.particles = newParticles


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        import util
        probDist = util.Counter()
        for particle in self.particles:
            probDist[particle] += 1

        for pos in probDist:
            probDist[pos] /= float(self.numParticles)

        return probDist



class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        cartesianProd = itertools.product(self.legalPositions, self.legalPositions)

        positions = []
        for item in cartesianProd:
            positions.append(item)

        flag = 0
        while flag == 0:
            for item in positions:
                if len(self.particles) == self.numParticles:
                    flag = 1
                    break
                self.particles.append(item)

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)
        
    def weighted_choice(self,choices, weights):
      import random
      """
      Return a randomly selected element from the `choices` list based on the given `weights`.

      Args:
          choices (list): List of elements to choose from.
          weights (list): List of weights corresponding to each element in `choices`.

      Returns:
          object: The randomly selected element.
      """
      total = sum(weights)
      threshold = random.uniform(0, total)
      cumulative_weight = 0

      for choice, weight in zip(choices, weights):
          cumulative_weight += weight
          if cumulative_weight >= threshold:
              return choice

      # Fallback to the last element if no choice was made (should not happen under normal circumstances)
      return choices[-1]
    
    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        pacmanPos = gameState.getPacmanPosition()
        numParticles = len(self.particles)
        weights = [1.0] * numParticles  # Initialize weights to 1.0 initially

        for j in range(numParticles):
            particle = self.particles[j]  # Get the j-th particle configuration (positions for all ghosts)
            jointObservationProb = 1.0  # Start with a probability of 1.0 for this particle

            for i in range(len(observation)):
                ghostPos = particle[i]  # Get the position of the i-th ghost in the current particle
                ghostJailPos = self.getJailPosition(i)
                observationProb = self.getObservationProb(observation[i], pacmanPos, ghostPos, ghostJailPos)
                jointObservationProb *= observationProb
            
            # Update the weight of the particle based on the joint observation probability
            weights[j] *= jointObservationProb

        # Normalize weights to form a probability distribution
        total_weight = sum(weights)
        if total_weight == 0:
            # Special case: All weights are zero, reinitialize particles uniformly
            self.initializeUniformly(gameState)
        else:
            normalized_weights = [w / total_weight for w in weights]

            # Resample particles based on the normalized weights using custom weighted sampling
            new_particles = []
            for _ in range(numParticles):
                chosen_particle = self.weighted_choice(self.particles, normalized_weights)
                new_particles.append(chosen_particle)

            # Update particles to the newly resampled particles
            self.particles = new_particles

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"# Create a copy of the current particle
            # Predict new positions for each ghost in the particle configuration
            prevGhostPositions = list(oldParticle)  # Previous positions of all ghosts in the particle
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                newGhostPos = newPosDist.sample()
                newParticle[i] = newGhostPos  # Update the position of the i-th ghost in the new particle# Convert to tuple and add to new particles list

            # Update particles to the newly p
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    import util
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState
  
class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
