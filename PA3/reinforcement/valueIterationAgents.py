# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        markov = self.mdp
        for k in range(self.iterations):
            vals_prime = util.Counter()
            for s in markov.getStates():
                act = self.getAction(s)
                if act is not None:
                    vals_prime[s] = self.getQValue(s, act)
            self.values = vals_prime

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        sigma = 0
        markov = self.mdp
        trans_prob_list = markov.getTransitionStatesAndProbs(state, action)
        for trans_prob in trans_prob_list:
            s_prime = trans_prob[0]
            prob = trans_prob[1]
            reward = markov.getReward(state, action, s_prime)
            val_prime = self.getValue(s_prime)
            sigma = sigma + prob * (reward + (self.discount * val_prime))

        return sigma
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        markov = self.mdp
        if markov.isTerminal(state):
            return None
        else:
            actions = markov.getPossibleActions(state)
            max_act = actions[0]
            max_val = self.getQValue(state, max_act)

            for a in actions:
                val = self.getQValue(state, a)
                if max_val <= val:
                    max_val = val
                    max_act = a

            return max_act

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        markov = self.mdp
        vals_prime = util.Counter()
        for i in range(self.iterations):
            states = markov.getStates()
            state_iteration = i % len(states)
            s = states[state_iteration]
            if markov.isTerminal(s):
                self.values[s] = 0
            act = self.getAction(s)
            if act is not None:
                vals_prime[s] = self.getQValue(s, act)
            self.values = vals_prime

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        markov = self.mdp
        predecessors = {}
        queue = util.PriorityQueue()

        for state in markov.getStates():
            predecessors[state] = set()

        for state in markov.getStates():
            q_vals = util.Counter()
            for action in markov.getPossibleActions(state):
                trans_prob = markov.getTransitionStatesAndProbs(state, action)
                for (trans_state, prob) in trans_prob:
                    if prob != 0:
                        predecessors[trans_state].add(state)
                q_vals[action] = self.computeQValueFromValues(state, action)

            if not markov.isTerminal(state):
                curr_val = self.values[state]
                max_q = q_vals.argMax()
                max_val = q_vals[max_q]
                diff = abs(curr_val - max_val)
                queue.update(state, -diff)

        for k in range(self.iterations):
            if queue.isEmpty():
                return

            s = queue.pop()
            if not markov.isTerminal(s):
                q_vals = util.Counter()
                for action in markov.getPossibleActions(s):
                    q_vals[action] = self.computeQValueFromValues(s, action)
                max_q = q_vals.argMax()
                self.values[s] = q_vals[max_q]

            for p in predecessors[s]:
                pred_vals = util.Counter()
                for act in markov.getPossibleActions(p):
                    pred_vals[act] = self.computeQValueFromValues(p, act)
                max_q = pred_vals.argMax()
                max_pred = pred_vals[max_q]
                diff = abs(self.values[p] - max_pred)

                if diff > self.theta:
                    queue.update(p, -diff)