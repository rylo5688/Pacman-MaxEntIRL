"""
This is an implementation of "Maximum Entropy Inverse Reinforcement Learning" (Ziebart et al., 2008)

Acknowledgement:
Matthew Alger's MaxEnt helped guide and influece this implementation
https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

"""

import numpy as np
import numpy.random as rn
import value_iteration
from itertools import product

def compute_svf(sample_paths, transition_probability, discount, policy):
    """
    This function is from Ziebart et al. 2008. It computes the expected state occupancy frequence using a techinique
    similar to the forward-backward algorithm for Conditional Random Fields or value iteration in Reinforcement Learning

    This code is directly from Mattew Alger's implementation of Ziebart's Algorithm 1 (Ziebart et al., 2008)
    """
    N_STATES, _, N_ACTIONS = np.shape(transition_probability)
    n_paths = sample_paths.shape[0]

    path_length = sample_paths.shape[1]

    # NOTE: Pacman always starts in the same spot
    start_state_counts = np.zeros(N_STATES)
    for path in sample_paths:
        start_state_counts[path[0, 0]] += 1
    p_start_state = start_state_counts/n_paths

    expected_svf = np.tile(p_start_state, (path_length, 1)).T # Creates an array that repeats p_start_state
    for t in range(1, path_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(N_STATES), range(N_STATES), range(N_ACTIONS)):
            expected_svf[k, t] += (expected_svf[i, t-1]  * policy[i][k] * transition_probability[i, j, k])
    return expected_svf.sum(axis=1)


def maxent_irl(sample_paths, feature_matrix, transition_probability, discount, iterations, learning_rate):
    """
    Find the reward function from the list of games (sample_paths)

    sample_paths: A list of paths. One path = one game
    feature_matrix: NxD matrix (N = number of states, D = Dimensionality of the state_
    transition_probability: NxNxA (N = number of states, A = Number of actions), each element contains P(next state | current state, action a)
    discount: Discount factor for the MDP
    iterations: Number of gradient descent steps
    learning_rate: Gradient descent rate

    -> Reward vector of size N
    """
    N_STATES, _, _ = np.shape(transition_probability)

    # Initialize the reward weights to random probabilities since we are going to adjust them as we look at the samples
    theta = rn.uniform(size=(feature_matrix.shape[1]))

    # Calculate feature expectations
    feature_expectations = np.zeros(feature_matrix.shape[1])
    for path in sample_paths:
        for state, _, _ in path:
            feature_expectations += feature_matrix[state]
    feature_expectations /= sample_paths.shape[0] # Divide each element by the total number of paths

    for _ in range(iterations):
        # 1. Solve for optimal policy w.r.t. rewards with value iteration
        rewards = feature_matrix.dot(theta) # Vector of reward values

        _, policy = value_iteration.ValueIterationAgent(rewards, discount=discount)

        # 2. Solve for state visitation frequences P(s | theta, T)
        svf = compute_svf(sample_paths, transition_probability, discount, policy)

        # 3. Compute gradient
        gradient = feature_expectations - feature_matrix.T.dot(svf)

        # 4. Update theta with one gradient step
        theta += learning_rate * gradient
    return feature_matrix.dot(theta).reshape((N_STATES, ))